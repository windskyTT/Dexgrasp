"""DirectRLEnv wrapper for first-step DemoGrasp Inspire checkpoint playback."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from Dexgrasp.assets.robots.fr3_inspire_tac_urdf import HAND_CFG

from .dexgrasp_env_cfg import DexgraspEnvCfg
from .observation import (
    load_object_point_cloud,
    make_policy_observation,
    quat_wxyz_to_xyzw,
    scale_to_joint_range,
    unscale_from_joint_range,
)
from .reference_motion import load_reference_motion


class DexgraspEnv(DirectRLEnv):
    """Minimal Isaac Lab environment for playing the DemoGrasp one-step Inspire policy."""

    cfg: DexgraspEnvCfg

    def __init__(self, cfg: DexgraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_hit_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_init_state_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._arm_joint_indices = self._find_joint_indices(HAND_CFG.arm_joint_names)
        self._active_hand_joint_indices = self._find_joint_indices(HAND_CFG.active_hand_joint_names)
        self._active_robot_joint_indices = [*self._arm_joint_indices, *self._active_hand_joint_indices]
        self._passive_mimic = self._build_passive_mimic_map()
        self._eef_body_idx = self._find_body_index(HAND_CFG.eef_link)

        joint_limits = self.robot.data.soft_joint_pos_limits[0]
        self._joint_lower_limits = joint_limits[:, 0]
        self._joint_upper_limits = joint_limits[:, 1]
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._previous_joint_targets = self._joint_targets.clone()

        self._object_points = load_object_point_cloud(
            self.cfg.object_urdf,
            HAND_CFG.num_obs_dict["objpcl"] // 3,
            self.device,
        )
        self._object_init_pose_xyzw = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self._object_init_pose_xyzw[:, 6] = 1.0
        self.reference_motion = load_reference_motion(
            self.cfg.reference_motion_path,
            self.num_envs,
            HAND_CFG.num_active_hand_dofs,
            self.device,
        )
        self._one_step_actions = torch.zeros_like(self._actions)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-self.cfg.clip_actions, self.cfg.clip_actions)

    def _apply_action(self) -> None:
        active_indices = self._active_robot_joint_indices
        active_targets = scale_to_joint_range(
            self._actions,
            self._joint_lower_limits[active_indices],
            self._joint_upper_limits[active_indices],
        )
        self._joint_targets[:, active_indices] = active_targets
        self._limit_target_step()
        self._apply_passive_mimic_targets()
        self._joint_targets = torch.max(
            torch.min(self._joint_targets, self._joint_upper_limits.unsqueeze(0)),
            self._joint_lower_limits.unsqueeze(0),
        )
        self.robot.set_joint_position_target(self._joint_targets)
        self._previous_joint_targets[:] = self._joint_targets

    def _get_observations(self) -> dict:
        eef_pos_w = self.robot.data.body_pos_w[:, self._eef_body_idx]
        eef_quat_wxyz = self.robot.data.body_quat_w[:, self._eef_body_idx]
        obs = make_policy_observation(
            eef_pos_w=eef_pos_w,
            eef_quat_wxyz=eef_quat_wxyz,
            object_init_pose_xyzw=self._object_init_pose_xyzw,
            object_points_local=self._object_points,
            object_pos_w=self.object.data.root_pos_w,
            clip=self.cfg.clip_observations,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.successes = (self.object.data.root_pos_w[:, 2] > 0.18).float()
        self.has_hit_table = self.robot.data.body_pos_w[:, self._eef_body_idx, 2] < 0.0
        return self.successes

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids = self._normalize_env_ids(env_ids)

        self.robot.reset(env_ids)
        self.object.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        object_root_state = self.object.data.default_root_state[env_ids].clone()
        object_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids)

        self._joint_targets[env_ids] = joint_pos
        self._previous_joint_targets[env_ids] = joint_pos
        self._actions[env_ids] = 0.0
        self._one_step_actions[env_ids] = 0.0
        self.successes[env_ids] = 0.0
        self.has_hit_table[env_ids] = False
        self.is_init_state_valid[env_ids] = True

        self._object_init_pose_xyzw[env_ids, :3] = object_root_state[:, :3]
        self._object_init_pose_xyzw[env_ids, 3:] = quat_wxyz_to_xyzw(object_root_state[:, 3:7])

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> dict:
        """Compatibility helper used by DemoGrasp's old PPO runner."""

        self._reset_idx(env_ids)
        observations = self._get_observations()
        return {"policy": observations["policy"], "obs": observations["policy"]}

    def get_state(self) -> torch.Tensor:
        """DemoGrasp PPO expects an asymmetric state tensor even when state_space is empty."""

        return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)

    def generate_reaching_plan_idx(self, env_ids: Sequence[int] | torch.Tensor, actions: torch.Tensor | None = None) -> None:
        """Store the one-step policy output before replaying the reference action sequence."""

        env_ids = self._normalize_env_ids(env_ids)
        if actions is None:
            self._one_step_actions[env_ids] = 0.0
            return
        self._one_step_actions[env_ids] = actions[env_ids].to(self.device).clamp(
            -self.cfg.clip_actions,
            self.cfg.clip_actions,
        )

    def compute_reference_actions(self) -> torch.Tensor:
        """Return a normalized joint-position action from the current reference timestep."""

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        ref_step = torch.clamp(self.episode_length_buf, max=self.reference_motion.num_steps - 1)
        action = torch.zeros_like(self._actions)

        arm_qpos = self.robot.data.default_joint_pos[:, self._arm_joint_indices]
        action[:, : len(self._arm_joint_indices)] = unscale_from_joint_range(
            arm_qpos,
            self._joint_lower_limits[self._arm_joint_indices],
            self._joint_upper_limits[self._arm_joint_indices],
        )

        hand_qpos = self.reference_motion.hand_qpos[env_ids, ref_step]
        hand_start = len(self._arm_joint_indices)
        action[:, hand_start : hand_start + len(self._active_hand_joint_indices)] = unscale_from_joint_range(
            hand_qpos,
            self._joint_lower_limits[self._active_hand_joint_indices],
            self._joint_upper_limits[self._active_hand_joint_indices],
        )
        return action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)

    def _find_joint_indices(self, joint_names: Sequence[str]) -> list[int]:
        indices: list[int] = []
        missing: list[str] = []
        for joint_name in joint_names:
            found, _ = self.robot.find_joints(joint_name)
            if len(found) == 0:
                missing.append(joint_name)
            else:
                indices.append(found[0])
        if missing:
            raise RuntimeError(f"Missing robot joints from DemoGrasp YAML: {missing}")
        return indices

    def _find_body_index(self, body_name: str) -> int:
        found, _ = self.robot.find_bodies(body_name)
        if len(found) == 0:
            raise RuntimeError(f"Missing robot body from DemoGrasp YAML: {body_name}")
        return found[0]

    def _build_passive_mimic_map(self) -> list[tuple[int, int, float]]:
        mimic_map: list[tuple[int, int, float]] = []
        for passive_name, spec in HAND_CFG.passive_joints.items():
            passive_idx = self._find_joint_indices([passive_name])[0]
            parent_idx = self._find_joint_indices([str(spec["mimic"])])[0]
            mimic_map.append((passive_idx, parent_idx, float(spec.get("multiplier", 1.0))))
        return mimic_map

    def _apply_passive_mimic_targets(self) -> None:
        for passive_idx, parent_idx, multiplier in self._passive_mimic:
            self._joint_targets[:, passive_idx] = self._joint_targets[:, parent_idx] * multiplier

    def _limit_target_step(self) -> None:
        arm_step = self.cfg.arm_action_speed * self.step_dt
        hand_step = self.cfg.hand_action_speed * self.step_dt
        self._joint_targets[:, self._arm_joint_indices] = torch.clamp(
            self._joint_targets[:, self._arm_joint_indices],
            self._previous_joint_targets[:, self._arm_joint_indices] - arm_step,
            self._previous_joint_targets[:, self._arm_joint_indices] + arm_step,
        )
        self._joint_targets[:, self._active_hand_joint_indices] = torch.clamp(
            self._joint_targets[:, self._active_hand_joint_indices],
            self._previous_joint_targets[:, self._active_hand_joint_indices] - hand_step,
            self._previous_joint_targets[:, self._active_hand_joint_indices] + hand_step,
        )

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self.robot._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
