from utils import get_pretrained_model, get_dataset, set_trainable_layers
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import h5py

class ACTPolicyWithIntermediate(nn.Module):
    """
    A wrapper around your ACTPolicy that returns the decoder embeddings
    (the final hidden states) *before* the final action_head.
    """
    def __init__(self, original_policy):
        super().__init__()
        self.original_policy = original_policy
        
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.original_policy.normalize_inputs(batch)
        if len(self.original_policy.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.original_policy.expected_image_keys],
                dim=-4
            )
        batch = self.original_policy.normalize_targets(batch)
        
        encoder_out, encoder_in_pos_embed, decoder_pos_embed = self.original_policy.model.forward_for_extracting_intermediate(batch)
        
        return encoder_out, encoder_in_pos_embed, decoder_pos_embed, batch["action"], batch["action_is_pad"]

def generate_embeddings_hdf5(
    policy_repo_id: str = "yanivmel1/new_dataset_cube_080000",
    training_data_repo_id: str = "yanivmel1/new_dataset_cube",
    output_hdf5_path: str = "train_embeddings.h5",
    pos_embed_path: str = "pos_embed.pt",
    batch_size: int = 8
):
    """
    Creates an HDF5 file, chunk by chunk, storing the encoder_out, labels, etc.
    Also saves the position embeddings once (pos_embed.pt).
    """
    # Get your policy and dataset
    policy = get_pretrained_model(policy_repo_id)
    policy = set_trainable_layers(policy)
    policy_for_embeddings = ACTPolicyWithIntermediate(policy).eval()

    dataset = get_dataset(training_data_repo_id)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_for_embeddings.to(device)

    # Before writing to HDF5, let's open it in "write" mode
    with h5py.File(output_hdf5_path, "w") as h5f:
        T = 602
        dim_model = 512
        A = 100
        action_dim = 6

        # Create datasets in HDF5 with initial shape=(0, ...) and maxshape=(None, ...)
        dset_encoder_out = h5f.create_dataset(
            "encoder_out",
            shape=(0, T, dim_model),
            maxshape=(None, T, dim_model),
            dtype='float32',
            chunks=True
        )
        dset_labels = h5f.create_dataset(
            "labels",
            shape=(0, A, action_dim),
            maxshape=(None, A, action_dim),
            dtype='float32',
            chunks=True
        )
        dset_action_is_pad = h5f.create_dataset(
            "action_is_pad",
            shape=(0, A),
            maxshape=(None, A),
            dtype='bool',
            chunks=True
        )

        saved_pos_embed = False
        encoder_in_pos_embed_to_save = None
        decoder_pos_embed_to_save = None

        idx = 0  # how many samples we've already written
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="Extracting embeddings"):
                for k, v in batch.items():
                    batch[k] = v.to(device)

                encoder_out, encoder_in_pos_embed, decoder_pos_embed, action, action_is_pad = \
                    policy_for_embeddings(batch)

                # encoder_out shape: [T, B, dim_model]
                # action shape: [B, A, action_dim or dim_model]
                # action_is_pad shape: [B, A]
                
                B = action.size(0)

                # Move them to CPU
                encoder_out = encoder_out.cpu().float()
                action = action.cpu().float()
                action_is_pad = action_is_pad.cpu()

                # Resize the HDF5 datasets to accommodate new samples
                dset_encoder_out.resize((idx + B, T, dim_model))
                dset_labels.resize((idx + B, A, action_dim))
                dset_action_is_pad.resize((idx + B, A))

                # Write them
                dset_encoder_out[idx:idx+B, :, :] = encoder_out.transpose(0, 1)  
                # note: HDF5 prefers (sample, T, dim), while your tensor is (T, B, dim).
                # We do transpose(0, 1) to get (B, T, dim).

                dset_labels[idx:idx+B, :, :] = action
                dset_action_is_pad[idx:idx+B, :] = action_is_pad

                idx += B

                # Save pos embeddings once
                if not saved_pos_embed:
                    # Save these to a separate .pt file
                    encoder_in_pos_embed_to_save = encoder_in_pos_embed.cpu()
                    # Some decoder_pos_embed might be a nn.Embedding or a tensor
                    if hasattr(decoder_pos_embed, "weight"):
                        decoder_pos_embed_to_save = decoder_pos_embed.weight.cpu()
                    else:
                        decoder_pos_embed_to_save = decoder_pos_embed.cpu()
                    saved_pos_embed = True
        
        # Done writing. Now save pos embeddings as a separate file:
        torch.save({
            "encoder_in_pos_embed": encoder_in_pos_embed_to_save, 
            "decoder_pos_embed": decoder_pos_embed_to_save
        }, pos_embed_path)
    
    print(f"\nFinished writing data to {output_hdf5_path} and position embeddings to {pos_embed_path}")


if __name__ == "__main__":

    generate_embeddings_hdf5()

    # policy_repo_id = "yanivmel1/new_dataset_cube_080000"
    # policy = get_pretrained_model(policy_repo_id)
    # policy = set_trainable_layers(policy)

    # # Wrap it
    # policy_for_embeddings = ACTPolicyWithIntermediate(policy)
    # policy_for_embeddings.eval()

    # # Load your dataset
    # training_data_repo_id = "yanivmel1/new_dataset_cube"
    # # val_set_repo_id = "yanivmel1/fine_tune_1"
    # dataset = get_dataset(training_data_repo_id)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # # Generate embeddings
    # all_encoder_out = []
    # all_labels = []
    # all_action_is_pad = []
    # policy_for_embeddings.to(device)

    # with torch.no_grad():
    #     for batch in tqdm.tqdm(data_loader):
    #         for k, v in batch.items():
    #             batch[k] = v.to(device)
    #         encoder_out, encoder_in_pos_embed, decoder_pos_embed, action, action_is_pad = policy_for_embeddings(batch)
    #         # Save them
    #         all_encoder_out.append(encoder_out.cpu()) # shape [T=602, B, dim_model]
    #         all_labels.append(action.cpu())  # shape [B, A=100, dim_model]
    #         all_action_is_pad.append(action_is_pad.cpu()) # shape [B, A=100]
    

    # all_encoder_out = torch.cat(all_encoder_out, dim=1) 
    # all_labels = torch.cat(all_labels, dim=0)  
    # all_action_is_pad = torch.cat(all_action_is_pad, dim=0)  
    # torch.save({"encoder_out": all_encoder_out, "labels": all_labels, "action_is_pad": all_action_is_pad}, "train_embeddings.pt")
    # # since all the pos embeddings are the same, we only need to save one of them
    # torch.save({"encoder_in_pos_embed": encoder_in_pos_embed.cpu(), "decoder_pos_embed": decoder_pos_embed.weight.cpu()}, "pos_embed.pt")
    
    # # Load the saved object
    # data = torch.load("train_embeddings.pt")

    # # Print the dimensions of each tensor
    # for key, value in data.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value.shape}")
    #     else:
    #         print(f"{key}: Not a tensor")
    
    # pos_embed = torch.load("pos_embed.pt")
    # print("Pos embeddings:")
    # for key, value in pos_embed.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value.shape}")
    #     else:
    #         print(f"{key}: Not a tensor")
    
