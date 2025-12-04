import math
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange
from typing import List, Dict
from tensordict import TensorDict, TensorDictBase


# Suppress datetime.utcnow() deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.augmentations import Augmenter
from src.nn import LAOM, ActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSInMemoryDataset,
    DCSLAOMInMemoryDataset,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    soft_update,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LAOMConfig:
    num_epochs: int = 150
    batch_size: int = 2048
    use_aug: bool = False   
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    latent_action_dim: int = 256
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    obs_head_dim: int = 512
    obs_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = False
    encoder_deep: bool = True
    target_tau: float = 0.01
    target_update_every: int = 1
    frame_stack: int = 3
    normalize: bool = True
    custom_dataset: bool = False
    has_policy_labels: bool = False
    has_distractor_labels: bool = False
    state_difference_probe: bool = False
    data_path: str = ''
    

@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = False
    frame_stack: int = 3
    data_path: str = "data/example-data.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class DecoderConfig:
    total_updates: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    hidden_dim: int = 128
    use_aug: bool = False
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class Config:
    project: str = "laom"
    group: str = "laom"
    name: str = "laom"
    seed: int = 0

    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def train_pla(config: LAOMConfig):
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

    lapo = LAOM(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
    ).to(DEVICE)

    target_lapo = deepcopy(lapo)
    for p in target_lapo.parameters():
        p.requires_grad_(False)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    if config.use_aug:
        augmenter = Augmenter(dataset.img_hw)

    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    state_difference_probe = nn.Linear(lapo.latent_act_dim, dataset.state_dim).to(DEVICE)
    state_difference_probe_optim = torch.optim.Adam(state_difference_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
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
            policy_labels = batch.get("policy_label", None)
            distractor_labels = batch.get("distractor_label", None)
            offset = batch["offset"]


            if config.use_aug:
                obs_aug = augmenter(obs)
                future_obs_aug = augmenter(future_obs)
                next_obs_aug = augmenter(next_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if config.use_aug:
                    # using augmenter directly will not work due to bf16
                    latent_next_obs, latent_action, obs_hidden = lapo(obs_aug, future_obs_aug)
                else:
                    latent_next_obs, latent_action, obs_hidden = lapo(obs, future_obs)

                with torch.no_grad():
                    if config.use_aug:
                        next_obs_target = target_lapo.encoder(next_obs_aug).flatten(1)
                    else:
                        next_obs_target = target_lapo.encoder(next_obs).flatten(1)

                loss = F.mse_loss(latent_next_obs, next_obs_target.detach())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            if i % config.target_update_every == 0:
                soft_update(target_lapo, lapo, tau=config.target_tau)

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, states)

                pred_state_difference = state_difference_probe(latent_action.detach())
                state_difference_probe_loss = F.mse_loss(pred_state_difference, state_diffs)

                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, actions)

                pred_state_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(pred_state_action, actions)

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

            if total_iterations % 100 == 0:
                wandb.log(
                    {
                        "lapo/mse_loss": loss.item(),
                        "lapo/state_probe_mse_loss": state_probe_loss.item(),
                        "lapo/state_difference_probe_mse_loss": state_difference_probe_loss.item(),
                        "lapo/act_probe_mse_loss": act_probe_loss.item(),
                        "lapo/state_act_probe_mse_loss": state_act_probe_loss.item(),
                        # "lapo/pos_diff_probe_mse_loss": pos_diff_probe_loss.item(),
                        # "lapo/vel_diff_probe_mse_loss": vel_diff_probe_loss.item(),
                        "lapo/throughput": total_tokens / (time.time() - start_time),
                        "lapo/learning_rate": scheduler.get_last_lr()[0],
                        "lapo/grad_norm": get_grad_norm(lapo).item(),
                        "lapo/target_obs_norm": torch.norm(next_obs_target, p=2, dim=-1).mean().item(),
                        "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                        "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                        "lapo/epoch": epoch,
                        "lapo/total_steps": total_iterations,
                    }
                )

    return lapo



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
    # stage 1: pretraining lapo on unlabeled dataset
    lapo = train_pla(config=config.lapo)


    run.finish()
    return lapo #, actor, action_decoder


if __name__ == "__main__":
    train()
