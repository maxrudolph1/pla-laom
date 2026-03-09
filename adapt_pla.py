import math
import os
import sys
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Set CUDA device before importing torch (respects CUDA_VISIBLE_DEVICES env var if set)
# Can override by setting CUDA_DEVICE environment variable
# if "CUDA_DEVICE" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pyrallis import field
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader
from tqdm import trange

# Suppress datetime.utcnow() deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device("cuda:1")


class AdapterMLP(nn.Module):
    """Adapter MLP that transforms latent actions from a frozen PLA model."""
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Final layer outputs same dimension as input (to keep latent action dim consistent)
        layers.append(nn.Linear(in_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class ReconstructionMLP(nn.Module):
    """Reconstruction MLP that tries to reconstruct the original latent action from adapted version.
    
    Optionally accepts a one-hot encoded background distractor vector as additional input.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.0, num_backgrounds: int = 0):
        super().__init__()
        self.num_backgrounds = num_backgrounds
        # Input dim includes latent action dim + one-hot background vector
        total_input_dim = input_dim + num_backgrounds
        
        layers = []
        in_dim = total_input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Final layer outputs same dimension as latent action (to reconstruct original latent action)
        layers.append(nn.Linear(in_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, background_onehot=None):
        """
        Args:
            x: Adapted latent action tensor [batch_size, latent_action_dim]
            background_onehot: One-hot encoded background labels [batch_size, num_backgrounds]
        """
        if self.num_backgrounds > 0 and background_onehot is not None:
            x = torch.cat([x, background_onehot], dim=-1)
        return self.mlp(x)


class AdaptedPLA(nn.Module):
    """
    Wrapper around a frozen PLA model with trainable adapter and reconstruction MLPs.
    
    Architecture:
        obs, next_obs -> frozen_pla.encoder -> obs_emb, next_obs_emb
        obs_emb, next_obs_emb -> frozen_pla.act_head (IDM) -> latent_action
        latent_action -> adapter_mlp -> adapted_latent_action
        adapted_latent_action, background_onehot -> reconstruction_mlp -> reconstructed_latent_action
        
    The reconstruction MLP is trained to reconstruct the original latent_action from adapted_latent_action,
    conditioned on the background distractor. This helps the adapter learn to remove background-specific
    information while preserving action-relevant structure.
    """
    def __init__(
        self,
        frozen_pla: nn.Module,
        adapter_hidden_dim: int = 512,
        adapter_num_layers: int = 2,
        adapter_dropout: float = 0.0,
        reconstruction_hidden_dim: int = 512,
        reconstruction_num_layers: int = 2,
        reconstruction_dropout: float = 0.0,
        num_backgrounds: int = 0,
    ):
        super().__init__()
        self.frozen_pla = frozen_pla
        self.num_backgrounds = num_backgrounds
        
        # Freeze all parameters of the original PLA model
        for param in self.frozen_pla.parameters():
            param.requires_grad_(False)
        
        # Get latent action dimension from the frozen model
        latent_action_dim = frozen_pla.latent_act_dim
        
        # Create adapter MLP (trainable)
        self.adapter_mlp = AdapterMLP(
            input_dim=latent_action_dim,
            hidden_dim=adapter_hidden_dim,
            num_layers=adapter_num_layers,
            dropout=adapter_dropout,
        )
        
        # Create reconstruction MLP (trainable) - takes adapted latent action + background one-hot
        self.reconstruction_mlp = ReconstructionMLP(
            input_dim=latent_action_dim,
            hidden_dim=reconstruction_hidden_dim,
            num_layers=reconstruction_num_layers,
            dropout=reconstruction_dropout,
            num_backgrounds=num_backgrounds,
        )
        
        # Expose properties from frozen PLA for compatibility
        self.latent_act_dim = latent_action_dim
        self.final_encoder_shape = frozen_pla.final_encoder_shape
        
    def forward(self, obs, next_obs, background_labels=None):
        """
        Forward pass through the adapted PLA.
        
        Args:
            obs: Current observation tensor
            next_obs: Next observation tensor
            background_labels: Background label indices [batch_size] (will be one-hot encoded)
        
        Returns:
            latent_next_obs: Output of the frozen obs_head (using adapted latent action)
            adapted_latent_action: Output of the adapter MLP
            obs_emb: Observation embedding from the frozen encoder
            original_latent_action: Original latent action from frozen IDM (for reconstruction loss)
            reconstructed_latent_action: Reconstructed latent action from reconstruction MLP
        """
        # Get embeddings from frozen encoder
        with torch.no_grad():
            obs_emb, next_obs_emb = self.frozen_pla.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
            # Get original latent action from frozen IDM
            original_latent_action = self.frozen_pla.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        
        # Pass through adapter MLP (trainable)
        adapted_latent_action = self.adapter_mlp(original_latent_action)
        
        # Create one-hot encoding for background labels if provided
        background_onehot = None
        if self.num_backgrounds > 0 and background_labels is not None:
            background_onehot = F.one_hot(background_labels, num_classes=self.num_backgrounds).float()
        
        # Reconstruct original latent action from adapted version + background info
        reconstructed_latent_action = self.reconstruction_mlp(adapted_latent_action, background_onehot)
        
        # Use frozen obs_head with adapted latent action
        with torch.no_grad():
            latent_next_obs = self.frozen_pla.obs_head(obs_emb.flatten(1).detach(), adapted_latent_action)
        
        return latent_next_obs, adapted_latent_action, obs_emb.detach(), original_latent_action, reconstructed_latent_action
    
    @torch.no_grad()
    def label(self, obs, next_obs):
        """Get adapted latent action for labeling."""
        obs_emb, next_obs_emb = self.frozen_pla.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        original_latent_action = self.frozen_pla.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        adapted_latent_action = self.adapter_mlp(original_latent_action)
        return adapted_latent_action
    
    def get_trainable_parameters(self):
        """Return only the trainable parameters (adapter and reconstruction MLPs)."""
        return list(self.adapter_mlp.parameters()) + list(self.reconstruction_mlp.parameters())
    
    def save(self, path):
        """Save the full model including frozen PLA and adapter/reconstruction MLPs."""
        torch.save(
            {
                "frozen_pla_state_dict": self.frozen_pla.state_dict(),
                "frozen_pla_init_kwargs": getattr(self.frozen_pla, "init_kwargs", None),
                "adapter_mlp_state_dict": self.adapter_mlp.state_dict(),
                "reconstruction_mlp_state_dict": self.reconstruction_mlp.state_dict(),
                "adapter_config": {
                    "adapter_hidden_dim": self.adapter_mlp.mlp[0].out_features if hasattr(self.adapter_mlp.mlp[0], 'out_features') else self.latent_act_dim,
                    "reconstruction_hidden_dim": self.reconstruction_mlp.mlp[0].out_features if hasattr(self.reconstruction_mlp.mlp[0], 'out_features') else self.latent_act_dim,
                    "num_backgrounds": self.num_backgrounds,
                },
            },
            path,
        )


@dataclass
class PLAConfig:
    # Training config
    num_epochs: int = 150
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    
    # Dataset config
    data_path: str = ''
    future_obs_offset: int = 1
    frame_stack: int = 1
    normalize: bool = True
    custom_dataset: bool = False
    
    # Latent action dim (should match pre-trained PLA)
    latent_action_dim: int = 256
    
    # Discriminator config
    discriminator_dim: int = 512
    num_discriminator_outputs: int = 12
    alignment_method: str = 'l1'
    
    # Adapter MLP config
    adapter_hidden_dim: int = 512
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.0
    
    # Reconstruction MLP config
    reconstruction_hidden_dim: int = 512
    reconstruction_num_layers: int = 2
    reconstruction_dropout: float = 0.0
    reconstruction_loss_coef: float = 1.0
    

@dataclass
class Config:
    project: str = "laom"
    group: str = "laom"
    name: str = "laom"
    save_path: str = 'artifacts'
    ckpt_path: str = ''  # Path to pre-trained PLA checkpoint (required)
    seed: int = 0

    pla: PLAConfig = field(default_factory=PLAConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def fine_tune_pla(pla: nn.Module, config: PLAConfig):
    """
    Fine-tune a PLA model by freezing it and training adapter/reconstruction MLPs.
    
    The frozen PLA is wrapped in an AdaptedPLA which adds:
    - An adapter MLP after the IDM (act_head) that transforms latent actions
    - A reconstruction MLP that tries to reconstruct the original latent action
    
    Only the adapter and reconstruction MLPs are trained.
    """
    dataset = DCSLAOMInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE, custom_dataset=config.custom_dataset, normalize=config.normalize
    )

    def td_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> TensorDictBase:
        stacked_dict = {}
        for key in batch[0].keys():
            stacked_dict[key] = torch.stack([item[key] for item in batch])
        return TensorDict(stacked_dict, batch_size=len(batch))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=td_collate_fn,
    )

    # Get number of backgrounds from dataset for reconstruction MLP conditioning
    num_backgrounds = dataset.get_num_discriminator_outputs()
    
    # Wrap the frozen PLA with adapter and reconstruction MLPs
    adapted_pla = AdaptedPLA(
        frozen_pla=pla,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_num_layers=config.adapter_num_layers,
        adapter_dropout=config.adapter_dropout,
        reconstruction_hidden_dim=config.reconstruction_hidden_dim,
        reconstruction_num_layers=config.reconstruction_num_layers,
        reconstruction_dropout=config.reconstruction_dropout,
        num_backgrounds=num_backgrounds,
    ).to(DEVICE)

    print("\n=== AdaptedPLA Architecture ===")
    print(f"Frozen PLA latent action dim: {adapted_pla.latent_act_dim}")
    print(f"Number of backgrounds: {num_backgrounds}")
    print(f"Adapter MLP: {adapted_pla.adapter_mlp}")
    print(f"Reconstruction MLP (input: latent_action + background_onehot): {adapted_pla.reconstruction_mlp}")
    print(f"Trainable parameters: {sum(p.numel() for p in adapted_pla.get_trainable_parameters())}")
    print(f"Frozen parameters: {sum(p.numel() for p in adapted_pla.frozen_pla.parameters())}")
    print("=" * 30 + "\n")

    # Only optimize the trainable parameters (adapter + reconstruction MLPs)
    optim = torch.optim.Adam(
        params=adapted_pla.get_trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    discriminator = Discriminator(config.latent_action_dim, discriminator_dim=config.discriminator_dim, num_outputs=dataset.get_num_discriminator_outputs()).to(DEVICE)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)

    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    state_act_linear_probe = nn.Linear(math.prod(adapted_pla.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    state_probe = nn.Linear(math.prod(adapted_pla.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    state_difference_probe = nn.Linear(adapted_pla.latent_act_dim, dataset.state_dim).to(DEVICE)
    state_difference_probe_optim = torch.optim.Adam(state_difference_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        adapted_pla.train()
        for i, batch in enumerate(dataloader):
            total_tokens += config.batch_size
            total_iterations += 1

            batch = batch.to(DEVICE)
            obs = normalize_img(batch["obs"].permute((0, 3, 1, 2)))
            next_obs = normalize_img(batch["next_obs"].permute((0, 3, 1, 2)))
            future_obs = normalize_img(batch["future_obs"].permute((0, 3, 1, 2)))
            actions = batch["action"]
            states = batch["state"]
            next_states = batch["next_state"]
            state_diffs = batch["state_diff"]
            background_labels = batch.get("background_labels", None)

            # Forward pass through adapted PLA (pass background labels for reconstruction conditioning)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                latent_next_obs, adapted_latent_action, obs_hidden, original_latent_action, reconstructed_latent_action = adapted_pla(obs, future_obs, background_labels)

                # Reconstruction loss: adapted_latent_action should be reconstructible back to original
                reconstruction_loss = F.mse_loss(reconstructed_latent_action, original_latent_action.detach())

                if config.alignment_method == 'discriminator':
                    discriminator_logits = discriminator(adapted_latent_action)
                    alignment_loss = F.cross_entropy(discriminator_logits, background_labels)
                elif config.alignment_method == 'contrastive':
                    import pdb; pdb.set_trace()
                elif config.alignment_method == 'l1':
                    alignment_loss = torch.norm(adapted_latent_action, p=1, dim=-1).mean()
                else:
                    alignment_loss = torch.tensor(0.0, device=DEVICE)
                
                # Total loss: alignment loss + reconstruction loss
                # Note: alignment_loss is negated because we want to maximize entropy/confusion
                loss = alignment_loss + config.reconstruction_loss_coef * reconstruction_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()

            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(adapted_pla.get_trainable_parameters(), max_norm=config.grad_norm)
                
            optim.step()
            scheduler.step()

            # Update probes using adapted latent actions
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, states)

                pred_state_difference = state_difference_probe(adapted_latent_action.detach())
                state_difference_probe_loss = F.mse_loss(pred_state_difference, state_diffs)

                pred_action = act_linear_probe(adapted_latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, actions)

                pred_state_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(pred_state_action, actions)

                discriminator_logits = discriminator(adapted_latent_action.detach())
                discriminator_loss = F.cross_entropy(discriminator_logits, background_labels)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            state_difference_probe_optim.zero_grad(set_to_none=True)
            state_difference_probe_loss.backward()
            state_difference_probe_optim.step()

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            discriminator_optim.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            discriminator_optim.step()

            if total_iterations % 100 == 0:
                # Compute gradient norm for trainable parameters only
                adapter_grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in adapted_pla.get_trainable_parameters() if p.grad is not None]),
                    2
                )
                wandb.log(
                    {
                        "adapted_pla/total_loss": loss.item(),
                        "adapted_pla/reconstruction_loss": reconstruction_loss.item(),
                        "adapted_pla/alignment_loss": alignment_loss.item(),
                        "adapted_pla/state_probe_mse_loss": state_probe_loss.item(),
                        "adapted_pla/state_difference_probe_mse_loss": state_difference_probe_loss.item(),
                        "adapted_pla/act_probe_mse_loss": act_probe_loss.item(),
                        "adapted_pla/state_act_probe_mse_loss": state_act_probe_loss.item(),
                        "adapted_pla/discriminator_loss": discriminator_loss.item(),
                        "adapted_pla/throughput": total_tokens / (time.time() - start_time),
                        "adapted_pla/learning_rate": scheduler.get_last_lr()[0],
                        "adapted_pla/adapter_grad_norm": adapter_grad_norm.item(),
                        "adapted_pla/adapted_latent_act_norm": torch.norm(adapted_latent_action, p=2, dim=-1).mean().item(),
                        "adapted_pla/original_latent_act_norm": torch.norm(original_latent_action, p=2, dim=-1).mean().item(),
                        "adapted_pla/epoch": epoch,
                        "adapted_pla/total_steps": total_iterations,
                    }
                )

    return adapted_pla



@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    set_seed(config.seed)

    # Load pre-trained PLA model from checkpoint
    if not config.ckpt_path:
        raise ValueError("ckpt_path is required - provide path to a pre-trained PLA checkpoint")
    print(f"Loading pre-trained PLA from: {config.ckpt_path}")
    pla = PLA.load(path=config.ckpt_path, map_location=DEVICE)
    print(f"Loaded PLA with latent_act_dim={pla.latent_act_dim}")
    
    # Fine-tune by freezing PLA and training adapter/reconstruction MLPs
    adapted_pla = fine_tune_pla(pla=pla, config=config.pla)
    artifact_path = Path(config.save_path) / config.group / config.name / "adapted"     
    os.makedirs(artifact_path, exist_ok=True)
    adapted_pla.save(artifact_path / "adapted_pla.pth")
    print(f"Saved adapted PLA to: {artifact_path / 'adapted_pla.pth'}")
    
    run.finish()
    return adapted_pla


if __name__ == "__main__":
    slurm = False
    if "--slurm" in sys.argv:
        slurm = True
        sys.argv.remove("--slurm")

    if slurm:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

        TEMPLATE="""#!/bin/bash

# Parameters
#SBATCH --account=amyzhang
#SBATCH --cpus-per-task=16
#SBATCH --error=slurm_scripts/job_%j/err.err
#SBATCH --output=slurm_scripts/job_%j/out.out
#SBATCH --job-name=pla-fine-tune
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --partition=allnodes
#SBATCH --exclude=slurm-node-008
#SBATCH --time=6:00:00

source /u/mrudolph/miniconda3/etc/profile.d/conda.sh
conda activate pla-laom
python """ + " ".join(sys.argv) + """
"""
        import subprocess

        path = "slurm_scripts/temp_submission.slurm"  # You may want to modify this value or get it from config

        # Open file and append a line
        with open(path, "w") as f:
            f.write(TEMPLATE)

        # Submit the file with sbatch
        result = subprocess.run(["sbatch", path], capture_output=True, text=True)
        jid = int(result.stdout.split(" ")[-1])
        print("sbatch output:", jid)
        print("sbatch error:", result.stderr)
        import os
        os.makedirs(f"slurm_scripts/job_{jid}", exist_ok=True)
        with open(f"slurm_scripts/job_{jid}/submission.sh", "w") as f:
            f.write(TEMPLATE)
    else:
        from src.nn import PLA, Discriminator
        from src.scheduler import linear_annealing_with_warmup
        from src.utils import (
            DCSLAOMInMemoryDataset,
            normalize_img,
            set_seed,
        )

        train()
