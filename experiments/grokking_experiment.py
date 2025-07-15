import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ACT_model_from_hf import ACTDecoder, ACTConfig
import os
from torch.cuda.amp import GradScaler
import time
from contextlib import nullcontext
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from tqdm import tqdm
from lerobot.common.datasets.utils import cycle
import matplotlib.pyplot as plt
from huggingface_hub import PyTorchModelHubMixin
import csv

class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, embed_path):
        data = torch.load(file_path)
           
        self.env_embeddings = data["encoder_out"]  # shape [num_of_tokens=602, num_of_samples, hidden_dim=512]
    
        self.action_embeddings = data["labels"] # shape [num_of_samples, num_of_actions=100, action_dim=6]
        self.action_is_pad = data["action_is_pad"] # shape [num_of_samples, num_of_actions=100]

        embed_data = torch.load(embed_path)
        self.pos_embeddings = embed_data["encoder_in_pos_embed"] # shape [num_of_tokens=602, 1, hidden_dim=512]        
        self.action_pos_embeddings = embed_data["decoder_pos_embed"] # shape [1, num_of_actions=100, hidden_dim=512]

        self.size = self.env_embeddings.size(1)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return {
            "embedding": self.env_embeddings[:, index, :],
            "pos_embedding": self.pos_embeddings,
            "action_pos_embedding": self.action_pos_embeddings,
            "action": self.action_embeddings[index, :, :],
            "action_is_pad": self.action_is_pad[index, :]
        }

class sub_model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = ACTDecoder(config) # the only thing that i want to tune here is the 
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])
    
    def set_empty_actions(self, batch_size, env_pos_embed):
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=env_pos_embed.dtype,
            device=env_pos_embed.device,
        )
        return decoder_in
    
    def forward(self, batch):
        env_embeddings = batch["embedding"].transpose(0, 1) # [602, 2, 512]
        env_pos_embed = batch["pos_embedding"][0,:] 
        action_pos_embeddings = batch["action_pos_embedding"][0,:]

        batch_size = env_embeddings.size(1)
        self.empty_actions = self.set_empty_actions(batch_size, env_pos_embed) # [100, 2, 512]

        decoder_out = self.decoder(self.empty_actions, 
                            env_embeddings, 
                            encoder_pos_embed=env_pos_embed, 
                            decoder_pos_embed=action_pos_embeddings.unsqueeze(1)) # [100 1, 512]
        
        decoder_out = decoder_out.transpose(0, 1) # Move back to (B, S, C)
        actions_hat = self.action_head(decoder_out) # 14
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        loss_dict = {"loss": l1_loss}
        return loss_dict
    

def get_embedding_dataloader(embed_dataset, batch_size=32, shuffle=True):
    loader = torch.utils.data.DataLoader(
        embed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    )
    return loader

def get_dataloader_iter(dataloader):
    dl_iter = cycle(dataloader)
    return dl_iter

def set_trainable_layers(policy):
    keep_grad = [
    "decoder.layers.0.linear1.weight",
    "decoder.layers.0.linear1.bias",
    "decoder.layers.0.linear2.weight",
    "decoder.layers.0.linear2.bias",
    "decoder.layers.0.norm3.weight",
    "decoder.layers.0.norm3.bias",
    "action_head.weight",
    "action_head.bias"
    ]
    for name, param in policy.named_parameters():
        if name not in keep_grad:  
            param.requires_grad = False

    for name, param in policy.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    return policy

def set_optimizer(policy, weight_decay):
    lr = 1e-5
    lr_backbone = 1e-5
    optimizer_params_dicts = [
        {"params": [p for n, p in policy.named_parameters()
                if not n.startswith("model.backbone") and p.requires_grad]},
        {"params": [p for n, p in policy.named_parameters()
                if n.startswith("model.backbone") and p.requires_grad],
                "lr": lr_backbone,},]
    optimizer = torch.optim.AdamW(optimizer_params_dicts, lr=lr, weight_decay=weight_decay)
    return optimizer

def print_num_params(policy):    
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    print(f"Number of learnable parameters: {num_learnable_params}")
    print(f"Number of total parameters: {num_total_params}")

def set_torch_cuda():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def log_train_info(info, step, dataset, train_batch_size):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]
    num_samples = (step + 1) * train_batch_size
    dataset_num_samples = len(dataset)
    num_epochs = num_samples / dataset_num_samples
    log_items = [
        f"step:{step}",
        # number of samples seen during training
        f"smpl:{num_samples}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # if not ~0, you are bottlenecked by cpu or io
    ]
    print(" ".join(log_items))
    return 

def update_policy(device, policy, batch, optimizer, lock=None):
    grad_clip_norm = 10
    lr_scheduler = None
    use_amp = False
    grad_scaler = GradScaler(enabled=use_amp)

    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
    grad_scaler.scale(loss).backward()

    # Unscale the graident of the optimzer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if isinstance(policy, PolicyWithUpdate):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
        **{k: v for k, v in output_dict.items() if k != "loss"},
    }
    info.update({k: v for k, v in output_dict.items() if k not in info})

    return info

def evaluate_policy(policy, val_dataloader, device):
    """
    Evaluate the policy on the validation set.
    Args:
        policy: The policy to evaluate.
        val_dataloader: DataLoader for the validation set.
        device: Device to run the evaluation on.
    Returns:
        Average loss over the validation set.
    """
    policy.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

            # Perform inference
            loss_dict = policy.forward(batch)
            l1_loss = loss_dict["loss"]
            print(f"l1_loss: {l1_loss}")
            total_loss += l1_loss.item() * batch["action"].size(0)
            total_samples += batch["action"].size(0)
            break

    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.3f}")
    return avg_loss

def save_losses_to_file(loss_log_file, step, train_loss, val_loss):
    if not os.path.exists(loss_log_file):
        with open(loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Training Loss", "Validation Loss"])  # Header row
    with open(loss_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([step, train_loss, val_loss])

if __name__ == "__main__":
    # data args
    train_embeddings_path = "train_embeddings.pt" #os.path.join("code", "train_embeddings.pt")
    val_embeddings_path = "val_embeddings.pt" #os.path.join("code", "train_embeddings.pt")
    pos_embed_path = "pos_embed.pt" #os.path.join("code", "pos_embed.pt") 
    batch_size_train = 8
    batch_size_val = 32

    # model args
    dim_feedforward = 3200 # default value in ACTConfig    
    # loss args
    weight_decay = 1e-4

    # training args
    offline_steps = 10000 
    log_freq = 100 
    save_freq = 1000 
    eval_freq = 1000 
    loss_log_file = "loss_log.csv"

    train_dataset = EmbeddingsDataset(train_embeddings_path, pos_embed_path)
    val_dataset = EmbeddingsDataset(val_embeddings_path, pos_embed_path)
    train_embeddings_loader = get_embedding_dataloader(train_dataset, batch_size_train)
    train_dataloader_iter = get_dataloader_iter(train_embeddings_loader)
    val_embeddings_loader = get_embedding_dataloader(val_dataset, batch_size_val)
    
    config = ACTConfig(n_obs_steps=1, chunk_size=100, n_action_steps=100, input_shapes={'observation.images.laptop': [3, 480, 640], 'observation.images.phone': [3, 480, 640], 'observation.state': [6]}, output_shapes={'action': [6]}, input_normalization_modes={'observation.images.laptop': 'mean_std', 'observation.images.phone': 'mean_std', 'observation.state': 'mean_std'}, output_normalization_modes={'action': 'mean_std'}, vision_backbone='resnet18', pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1', replace_final_stride_with_dilation=False, pre_norm=False, dim_model=512, n_heads=8, dim_feedforward=3200, feedforward_activation='relu', n_encoder_layers=4, n_decoder_layers=1, use_vae=True, latent_dim=32, n_vae_encoder_layers=4, temporal_ensemble_coeff=None, dropout=0.1, kl_weight=10.0)
    config.dim_feedforward = dim_feedforward
    policy = sub_model(config)
    policy = set_trainable_layers(policy)
    optimizer = set_optimizer(policy, weight_decay) 
    print_num_params(policy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_torch_cuda()
    policy.to(device)
    policy.train()

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []
    steps = []
    
    step = 0  # number of policy updates (forward + backward + optim)
    offline_step = 0

    for _ in tqdm(range(step, offline_steps)):
        start_time = time.perf_counter()
        batch = next(train_dataloader_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(device, policy, batch, optimizer) # train_info["loss"] is float which is the loss of the batch after one step
        train_step_loss = train_info["loss"]
        train_info["dataloading_s"] = dataloading_s
        train_losses.append(train_step_loss)  # Store training loss
        steps.append(step)

        if (step % eval_freq == 0) or step == 0:
            avg_loss_val = evaluate_policy(policy, val_embeddings_loader, device)  # loss is average loss over all the validation set, new value after evry eval_freq steps
            val_losses.append(avg_loss_val)  # Store validation loss
            print(f"Step {step}: Validation Loss = {avg_loss_val:.4f}")
        else:
            val_losses.append(None)  # Append None for alignment

        if step % log_freq == 0:
            log_train_info(train_info, step, train_dataset, batch_size_train) 
        
        save_losses_to_file(loss_log_file, step, train_step_loss, val_losses[-1])
            
        if step % save_freq == 0  or step == offline_steps:
            step_identifier = f"{step:0{5}d}" 
            save_dir = os.path.join("checkpoints", step_identifier)
            policy.save_pretrained(save_dir)
            print("Resume training")

        step += 1
        offline_step += 1 


    # Plotting after training
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", color="blue", marker="o", linestyle="-")
    plt.plot([s for s, v in zip(steps, val_losses) if v is not None], [v for v in val_losses if v is not None],
            label="Validation Loss", color="orange", marker="x", linestyle="--")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("training_validation_loss.png")
    plt.show()
