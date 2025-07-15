from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import cycle
import torch
import os
from huggingface_hub import snapshot_download
from ACT_model_from_hf import ACTPolicy
from torch.cuda.amp import GradScaler
import time
from contextlib import nullcontext
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_delta_timestamps(fps, chunk_size):
    fps_value = fps
    delta_timestamps = {
        "action": [i / fps_value for i in range(chunk_size)]
    }
    return delta_timestamps

def modify_to_resnet_stats(dataset):
    # Define common statistics for both image sources
    common_stats = {
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]]
    }
    keys = ["observation.images.laptop", "observation.images.phone"]
    for key in keys:
        for stats_type, value in common_stats.items():
            dataset.stats[key][stats_type] = torch.tensor(value, dtype=torch.float32)

def get_dataset(repo_id):
    fps = 30
    chunk_size = 100
    image_transforms = None 
    video_backend = 'pyav'
    split='train'
    delta_timestamps = create_delta_timestamps(fps, chunk_size)
    dataset = LeRobotDataset(repo_id, split=split, delta_timestamps=delta_timestamps, image_transforms=image_transforms, video_backend=video_backend)
    modify_to_resnet_stats(dataset)
    return dataset

def get_dataloader(dataset, batch_size, device):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    return dataloader

def get_dataloader_iter(dataloader):
    dl_iter = cycle(dataloader)
    return dl_iter

def load_pretrained_model_from_hf(repo_id):
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    model_name = repo_id.split('/')[-1]
    local_dir = os.path.join(parent_dir, "models", model_name)
    # snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return local_dir

def get_pretrained_model(repo_id):
    model_dir = load_pretrained_model_from_hf(repo_id)
    policy = ACTPolicy.from_pretrained(model_dir)
    return policy

def set_trainable_layers(policy):
    keep_grad = [
    "model.decoder.layers.0.linear1.weight",
    "model.decoder.layers.0.linear1.bias",
    "model.decoder.layers.0.linear2.weight",
    "model.decoder.layers.0.linear2.bias",
    "model.decoder.layers.0.norm3.weight",
    "model.decoder.layers.0.norm3.bias",
    "model.action_head.weight",
    "model.action_head.bias"
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

def log_train_info(info, step, dataset):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]
    num_samples = (step + 1) * batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{step}",
        # number of samples seen during training
        f"smpl:{num_samples}",
        # number of episodes seen during training
        f"ep:{num_episodes}",
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



if __name__ == "__main__":

    repo_id = "yanivmel1/new_dataset_cube_080000"
    training_data_repo_id = "yanivmel1/new_dataset_cube"
    val_set_repo_id = "yanivmel1/fine_tune_1"
    batch_size = 8
    batch_size_val = 32
    step = 0  # number of policy updates (forward + backward + optim)
    offline_steps = 5 # 80000
    offline_step = 0
    log_freq = 5 # 100
    save_freq = 10 # 10000
    weight_decay = 1e-4
    eval_freq = 2 # 100

    policy = get_pretrained_model(repo_id)
    policy = set_trainable_layers(policy)
    optimizer = set_optimizer(policy, weight_decay) 
    print_num_params(policy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(training_data_repo_id)
    data_loader = get_dataloader(dataset, batch_size, device)
    dl_iter = get_dataloader_iter(data_loader)

    val_dataset = get_dataset(val_set_repo_id)
    val_dataloader = get_dataloader(val_dataset, batch_size_val, device)

    set_torch_cuda()
    policy.to(device)
    policy.train()

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []
    steps = []

    for _ in tqdm(range(step, offline_steps)):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(device, policy, batch, optimizer) # train_info["loss"] is float which is the loss of the batch after one step
        train_step_loss = train_info["loss"]
        train_info["dataloading_s"] = dataloading_s
        train_losses.append(train_step_loss)  # Store training loss
        steps.append(step)

        if step % eval_freq == 0:
            avg_loss_val = evaluate_policy(policy, val_dataloader, device)  # loss is average loss over all the validation set, new value after evry eval_freq steps
            val_losses.append(avg_loss_val)  # Store validation loss
            print(f"Step {step}: Validation Loss = {avg_loss_val:.4f}")
        else:
            val_losses.append(None)  # Append None for alignment

        if step % log_freq == 0:
            log_train_info(train_info, step, dataset) 
            
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
