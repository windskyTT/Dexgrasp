"""Temporary FR3 + Inspire Tac robot config for migration step 1.

This file intentionally keeps the URDF/YAML source of truth visible.  Step 2 can
replace this with a cleaned Isaac Lab USD/Python asset after the checkpoint play
path is proven.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from Dexgrasp.utils.demograsp_yaml import load_hand_config

HAND_CFG = load_hand_config()
REPO_ROOT = Path(__file__).resolve().parents[5]
TMP_USD_DIR = Path(__file__).resolve().parent / "_tmp_usd" / "fr3_inspire_tac"
TMP_USD_PATH = TMP_USD_DIR / "fr3_inspire_tac.usd"
REFERENCE_MOTION_PATH = (
    Path(__file__).resolve().parents[2] / "tasks" / "direct" / "dexgrasp" / "reference" / "grasp_ref_inspire.pkl"
)
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "ppo_onestep" / "inspire.pt"
OBJECT_ROOT = REPO_ROOT / "assets" / "objects"
DEFAULT_OBJECT_LIST = OBJECT_ROOT / "union_ycb_unidex" / "union_ycb_debugset.yaml"
DEFAULT_OBJECT_URDF = OBJECT_ROOT / "union_ycb_unidex" / "urdf" / "006_mustard_bottle.urdf"


FR3_INSPIRE_TAC_URDF_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(HAND_CFG.urdf_path),
        usd_dir=str(TMP_USD_DIR),
        usd_file_name=TMP_USD_PATH.name,
        force_usd_conversion=False,
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.0, damping=4.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=HAND_CFG.default_joint_positions,
    ),
    actuators={
        "fr3_arm": ImplicitActuatorCfg(
            joint_names_expr=HAND_CFG.arm_joint_names,
            effort_limit_sim=87.0,
            velocity_limit_sim=3.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "inspire_active_hand": ImplicitActuatorCfg(
            joint_names_expr=HAND_CFG.active_hand_joint_names,
            effort_limit_sim=12.0,
            velocity_limit_sim=6.28,
            stiffness=80.0,
            damping=4.0,
        ),
        "inspire_passive_hand": ImplicitActuatorCfg(
            joint_names_expr=sorted(HAND_CFG.passive_joints),
            effort_limit_sim=4.0,
            velocity_limit_sim=6.28,
            stiffness=20.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Temporary Isaac Lab articulation config backed by the original DemoGrasp URDF."""
