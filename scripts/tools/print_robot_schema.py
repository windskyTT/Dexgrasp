"""Launch the Dexgrasp task and print the robot schema seen by Isaac Lab."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print FR3+Inspire joint/body schema from Isaac Lab.")
parser.add_argument("--task", type=str, default="DexGrasp-Inspire-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import Dexgrasp.tasks  # noqa: F401
from Dexgrasp.assets.robots.fr3_inspire_tac_urdf import HAND_CFG


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.robot

    print("joint_names:")
    for index, name in enumerate(robot.joint_names):
        print(f"  {index:02d}: {name}")

    print("body_names:")
    for index, name in enumerate(robot.body_names):
        print(f"  {index:02d}: {name}")

    print("arm joint indices:")
    print(env.unwrapped._arm_joint_indices)
    print("active hand joint indices:")
    print(env.unwrapped._active_hand_joint_indices)
    print("default joint positions:")
    print(robot.data.default_joint_pos[0].detach().cpu().tolist())
    print("yaml arm joints:")
    print(HAND_CFG.arm_joint_names)
    print("yaml active hand joints:")
    print(HAND_CFG.active_hand_joint_names)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

