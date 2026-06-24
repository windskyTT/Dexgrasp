"""Reference-motion helpers for DemoGrasp one-step policy playback."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ReferenceMotion:
    """Reference wrist and hand trajectory expanded to all Isaac Lab environments."""

    wrist_initobj_pos: torch.Tensor
    wrist_quat_xyzw: torch.Tensor
    hand_qpos: torch.Tensor

    @property
    def num_steps(self) -> int:
        return int(self.hand_qpos.shape[1])


def _expand_sequence(value, num_envs: int, device: torch.device, last_dim: int | None = None) -> torch.Tensor:
    """Convert a pickle value into [num_envs, T, D], repeating a single demo when needed."""

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.view(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if last_dim is not None and tensor.shape[-1] > last_dim:
        tensor = tensor[..., :last_dim]
    if tensor.shape[0] == num_envs:
        return tensor
    if tensor.shape[0] > num_envs:
        return tensor[:num_envs]
    return tensor[:1].repeat(num_envs, 1, 1)


def _fallback_reference(num_envs: int, num_hand_dofs: int, device: torch.device, num_steps: int = 50) -> ReferenceMotion:
    """Keep the environment runnable even when the legacy pickle cannot be loaded."""

    wrist_pos = torch.zeros((num_envs, num_steps, 3), dtype=torch.float32, device=device)
    wrist_quat = torch.zeros((num_envs, num_steps, 4), dtype=torch.float32, device=device)
    wrist_quat[..., 3] = 1.0
    hand_qpos = torch.zeros((num_envs, num_steps, num_hand_dofs), dtype=torch.float32, device=device)
    return ReferenceMotion(wrist_pos, wrist_quat, hand_qpos)


def load_reference_motion(
    path: str | Path,
    num_envs: int,
    num_hand_dofs: int,
    device: torch.device,
) -> ReferenceMotion:
    """Load DemoGrasp's `grasp_ref_inspire.pkl` and normalize its tensor shapes."""

    path = Path(path)
    if not path.exists():
        return _fallback_reference(num_envs, num_hand_dofs, device)

    try:
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        wrist_pos = _expand_sequence(raw["wrist_initobj_pos"], num_envs, device, last_dim=3)
        wrist_quat = _expand_sequence(raw["wrist_quat"], num_envs, device, last_dim=4)
        hand_qpos = _expand_sequence(raw["hand_qpos"], num_envs, device, last_dim=num_hand_dofs)
    except Exception:
        return _fallback_reference(num_envs, num_hand_dofs, device)

    steps = min(wrist_pos.shape[1], wrist_quat.shape[1], hand_qpos.shape[1])
    return ReferenceMotion(wrist_pos[:, :steps], wrist_quat[:, :steps], hand_qpos[:, :steps])

