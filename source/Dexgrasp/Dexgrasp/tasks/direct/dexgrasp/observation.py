"""Observation helpers for the temporary DemoGrasp-compatible play environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def scale_to_joint_range(actions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized actions in [-1, 1] to joint position targets."""

    return 0.5 * (actions + 1.0) * (upper - lower) + lower


def unscale_from_joint_range(values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map joint positions back to normalized [-1, 1] actions."""

    return (2.0 * values - upper - lower) / (upper - lower).clamp(min=1.0e-6)


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Isaac Lab uses wxyz quaternions; DemoGrasp policy observations used xyzw order."""

    return torch.cat((quat[..., 1:], quat[..., :1]), dim=-1)


def load_object_point_cloud(object_urdf: str | Path, num_points: int, device: torch.device) -> torch.Tensor:
    """Load the debug object's pre-sampled point cloud or return zeros if it is unavailable."""

    urdf_path = Path(object_urdf)
    pointcloud_path = urdf_path.parents[1] / "pointclouds" / f"{urdf_path.stem}.npy"
    if not pointcloud_path.exists():
        return torch.zeros((num_points, 3), dtype=torch.float32, device=device)

    points = torch.as_tensor(np.load(pointcloud_path), dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 3:
        return torch.zeros((num_points, 3), dtype=torch.float32, device=device)
    if points.shape[0] >= num_points:
        return points[:num_points]

    pad = torch.zeros((num_points - points.shape[0], 3), dtype=torch.float32, device=device)
    return torch.cat((points, pad), dim=0)


def make_policy_observation(
    eef_pos_w: torch.Tensor,
    eef_quat_wxyz: torch.Tensor,
    object_init_pose_xyzw: torch.Tensor,
    object_points_local: torch.Tensor,
    object_pos_w: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Build the step-1 policy observation: eefpose + objinitpose + objpcl."""

    num_envs = eef_pos_w.shape[0]
    eef_pose_xyzw = torch.cat((eef_pos_w, quat_wxyz_to_xyzw(eef_quat_wxyz)), dim=-1)
    points_w = object_points_local.unsqueeze(0).repeat(num_envs, 1, 1) + object_pos_w.unsqueeze(1)
    obs = torch.cat((eef_pose_xyzw, object_init_pose_xyzw, points_w.reshape(num_envs, -1)), dim=-1)
    return torch.clamp(obs, -clip, clip)

