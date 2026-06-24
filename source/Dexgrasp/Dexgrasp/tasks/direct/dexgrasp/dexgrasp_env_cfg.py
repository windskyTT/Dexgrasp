"""Configuration for the first-step FR3 + Inspire Tac one-step play environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from Dexgrasp.assets.robots.fr3_inspire_tac_urdf import (
    CHECKPOINT_PATH,
    DEFAULT_OBJECT_LIST,
    DEFAULT_OBJECT_URDF,
    FR3_INSPIRE_TAC_URDF_CFG,
    HAND_CFG,
    REFERENCE_MOTION_PATH,
)


@configclass
class DexgraspEnvCfg(DirectRLEnvCfg):
    """Small GUI-first environment used to prove the migrated Inspire checkpoint path."""

    decimation = 2
    episode_length_s = 50.0 / 60.0
    action_space = HAND_CFG.num_actions
    observation_space = HAND_CFG.observation_dim("eefpose+objinitpose+objpcl")
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    robot_cfg: ArticulationCfg = FR3_INSPIRE_TAC_URDF_CFG
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(DEFAULT_OBJECT_URDF),
            fix_base=False,
            merge_fixed_joints=False,
            collision_from_visuals=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.1, 0.08), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    hand_yaml_path = str(HAND_CFG.yaml_path)
    checkpoint_path = str(CHECKPOINT_PATH)
    reference_motion_path = str(REFERENCE_MOTION_PATH)
    object_list = str(DEFAULT_OBJECT_LIST)
    object_urdf = str(DEFAULT_OBJECT_URDF)
    observation_type = "eefpose+objinitpose+objpcl"

    clip_actions = 1.0
    clip_observations = 5.0
    arm_action_speed = 1.57
    hand_action_speed = 6.28
    tracking_reference_lift_timestep = 13
    randomize_grasp_pose_range = 1.0
