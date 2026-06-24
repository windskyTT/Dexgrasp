"""Inspect the migrated Inspire one-step checkpoint against the Dexgrasp env dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Dexgrasp.algo.ppo_onestep import ActorCritic
from Dexgrasp.tasks.direct.dexgrasp.dexgrasp_env_cfg import DexgraspEnvCfg


DEFAULT_MODEL_CFG = {
    "backbone_type": "pn",
    "freeze_backbone": False,
    "pi_hid_sizes": [1024, 1024, 512, 512],
    "vf_hid_sizes": [1024, 1024, 512, 512],
    "activation": "elu",
    "pc_shape": [512, 3],
    "pc_emb_dim": 128,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Inspire checkpoint loading and policy dimensions.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo_onestep/inspire.pt"))
    parser.add_argument("--task", type=str, default="DexGrasp-Inspire-Direct-v0")
    parser.add_argument("--use_pcl_backbone", action="store_true")
    args = parser.parse_args()

    env_cfg = DexgraspEnvCfg()
    obs_dim = int(env_cfg.observation_space)
    action_dim = int(env_cfg.action_space)
    print(f"task: {args.task}")
    print(f"env observation dim: {obs_dim}")
    print(f"env action dim: {action_dim}")
    print(f"checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected state_dict checkpoint, got {type(checkpoint)!r}")
    print(f"checkpoint keys: {len(checkpoint)}")
    for index, (name, value) in enumerate(checkpoint.items()):
        if index >= 12:
            break
        print(f"  {name}: {tuple(value.shape) if hasattr(value, 'shape') else type(value)}")

    model = ActorCritic(
        obs_shape=(obs_dim,),
        states_shape=(0,),
        actions_shape=(action_dim,),
        initial_std=0.8,
        model_cfg=DEFAULT_MODEL_CFG,
        asymmetric=False,
        use_pcl=args.use_pcl_backbone,
    )
    model.load_state_dict(checkpoint, strict=True)
    print("strict load: OK")

    sample_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    action = model(sample_obs, inference=True)
    print(f"policy output shape: {tuple(action.shape)}")
    if action.shape[-1] != action_dim:
        raise RuntimeError(f"Policy output dim {action.shape[-1]} != env action dim {action_dim}")


if __name__ == "__main__":
    main()

