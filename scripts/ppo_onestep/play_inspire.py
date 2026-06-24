"""Play the migrated DemoGrasp Inspire one-step checkpoint in Isaac Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play inspire.pt on DexGrasp-Inspire-Direct-v0.")
parser.add_argument("--task", type=str, default="DexGrasp-Inspire-Direct-v0")
parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo_onestep/inspire.pt"))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--object_list", type=str, default="union_ycb_debugset")
parser.add_argument("--use_pcl_backbone", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import Dexgrasp.tasks  # noqa: F401
from Dexgrasp.algo.ppo_onestep import ActorCritic


DEFAULT_MODEL_CFG = {
    "backbone_type": "pn",
    "freeze_backbone": False,
    "pi_hid_sizes": [1024, 1024, 512, 512],
    "vf_hid_sizes": [1024, 1024, 512, 512],
    "activation": "elu",
    "pc_shape": [512, 3],
    "pc_emb_dim": 128,
}


def _policy_obs(reset_or_step_result) -> torch.Tensor:
    obs = reset_or_step_result[0] if isinstance(reset_or_step_result, tuple) else reset_or_step_result
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs_dim = int(base_env.cfg.observation_space)
    action_dim = int(base_env.cfg.action_space)
    actor_critic = ActorCritic(
        obs_shape=(obs_dim,),
        states_shape=(0,),
        actions_shape=(action_dim,),
        initial_std=0.8,
        model_cfg=DEFAULT_MODEL_CFG,
        asymmetric=False,
        use_pcl=args_cli.use_pcl_backbone,
    ).to(base_env.device)
    actor_critic.load_state_dict(torch.load(args_cli.checkpoint, map_location=base_env.device), strict=True)
    actor_critic.eval()

    reset_result = env.reset()
    obs = _policy_obs(reset_result)
    env_ids = torch.arange(base_env.num_envs, dtype=torch.long, device=base_env.device)

    with torch.inference_mode():
        plan_actions = actor_critic(obs, inference=True)
        base_env.generate_reaching_plan_idx(env_ids, actions=plan_actions)

        while simulation_app.is_running():
            actions = base_env.compute_reference_actions()
            step_result = env.step(actions)
            obs = _policy_obs(step_result)

            done = None
            if isinstance(step_result, tuple) and len(step_result) == 5:
                done = step_result[2] | step_result[3]
            if done is not None and bool(torch.as_tensor(done, device=base_env.device).all()):
                print(f"success rate: {base_env.successes.float().mean().item():.3f}")
                reset_result = env.reset()
                obs = _policy_obs(reset_result)
                plan_actions = actor_critic(obs, inference=True)
                base_env.generate_reaching_plan_idx(env_ids, actions=plan_actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

