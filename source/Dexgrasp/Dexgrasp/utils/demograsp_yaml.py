"""Read the temporary DemoGrasp YAML/URDF metadata used during migration step 1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HAND_YAML = PACKAGE_DIR / "hand" / "fr3_inspire_tac.yaml"


@dataclass(frozen=True)
class JointLimit:
    """Joint limit parsed from the DemoGrasp URDF."""

    lower: float
    upper: float
    effort: float | None
    velocity: float | None


@dataclass(frozen=True)
class DemoGraspHandConfig:
    """Small, explicit subset of the DemoGrasp hand YAML needed by step 1."""

    name: str
    yaml_path: Path
    urdf_path: Path
    visual_urdf_path: Path | None
    arm_joint_names: list[str]
    active_hand_joint_names: list[str]
    passive_joints: dict[str, dict[str, float | str]]
    default_joint_positions: dict[str, float]
    all_joint_names: list[str]
    joint_limits: dict[str, JointLimit]
    palm_link: str
    eef_link: str
    fingertip_links: list[str]
    num_actions: int
    num_obs_dict: dict[str, int]

    @property
    def active_robot_joint_names(self) -> list[str]:
        """Policy-controlled joints: 7 arm joints followed by 6 Inspire active hand joints."""

        return [*self.arm_joint_names, *self.active_hand_joint_names]

    @property
    def num_active_hand_dofs(self) -> int:
        return len(self.active_hand_joint_names)

    def observation_dim(self, obs_type: str) -> int:
        """Return the observation dimension for a DemoGrasp observationType string."""

        requested = set(obs_type.split("+"))
        return sum(dim for name, dim in self.num_obs_dict.items() if name in requested)


def _resolve_robot_asset(path_from_yaml: str) -> Path:
    """DemoGrasp YAML paths are relative to the old assets root; Dexgrasp keeps them under assets/robots."""

    return PACKAGE_DIR / "assets" / "robots" / path_from_yaml


def _parse_joint_limits(urdf_path: Path) -> tuple[list[str], dict[str, JointLimit]]:
    """Parse URDF joint order and limits without depending on Isaac Sim being launched."""

    root = ET.parse(urdf_path).getroot()
    joint_names: list[str] = []
    limits: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        joint_name = joint.attrib.get("name", "")
        if not joint_name or joint_type == "fixed":
            continue
        joint_names.append(joint_name)
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint_name] = JointLimit(
            lower=float(limit.attrib.get("lower", "0.0")),
            upper=float(limit.attrib.get("upper", "0.0")),
            effort=float(limit.attrib["effort"]) if "effort" in limit.attrib else None,
            velocity=float(limit.attrib["velocity"]) if "velocity" in limit.attrib else None,
        )
    return joint_names, limits


def load_hand_config(path: str | Path = DEFAULT_HAND_YAML) -> DemoGraspHandConfig:
    """Load DemoGrasp's hand YAML and the matching URDF joint schema."""

    yaml_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(yaml_path.read_text())
    urdf_path = _resolve_robot_asset(raw["robotAssetFile"]).resolve()
    visual_urdf = raw.get("robotAssetFileVisualRealistic")
    visual_urdf_path = _resolve_robot_asset(visual_urdf).resolve() if visual_urdf else None
    all_joint_names, joint_limits = _parse_joint_limits(urdf_path)

    default_values = list(raw["default_dof_pos"])
    default_joint_positions = {
        joint_name: float(default_values[index])
        for index, joint_name in enumerate(all_joint_names[: len(default_values)])
    }

    return DemoGraspHandConfig(
        name=raw["name"],
        yaml_path=yaml_path,
        urdf_path=urdf_path,
        visual_urdf_path=visual_urdf_path,
        arm_joint_names=list(raw["arm_dof_names"]),
        active_hand_joint_names=list(raw["hardware_active_hand_dof_names"]),
        passive_joints=dict(raw.get("passive_joints", {})),
        default_joint_positions=default_joint_positions,
        all_joint_names=all_joint_names,
        joint_limits=joint_limits,
        palm_link=raw["palm_link"],
        eef_link=raw["eef_link"],
        fingertip_links=list(raw["fingertips_link"]),
        num_actions=int(raw["numActions"]),
        num_obs_dict={name: int(dim) for name, dim in raw["num_obs_dict"].items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DemoGrasp hand YAML and matching URDF joint schema.")
    parser.add_argument("--file", type=Path, default=DEFAULT_HAND_YAML, help="Path to fr3_inspire_tac.yaml.")
    parser.add_argument(
        "--obs_type",
        type=str,
        default="eefpose+objinitpose+objpcl",
        help="DemoGrasp observationType string to size.",
    )
    args = parser.parse_args()

    cfg = load_hand_config(args.file)
    summary = {
        "name": cfg.name,
        "yaml_path": str(cfg.yaml_path),
        "urdf_path": str(cfg.urdf_path),
        "arm_joint_names": cfg.arm_joint_names,
        "active_hand_joint_names": cfg.active_hand_joint_names,
        "passive_joint_names": sorted(cfg.passive_joints),
        "all_joint_names": cfg.all_joint_names,
        "default_joint_positions": cfg.default_joint_positions,
        "num_actions": cfg.num_actions,
        "observation_type": args.obs_type,
        "observation_dim": cfg.observation_dim(args.obs_type),
        "eef_link": cfg.eef_link,
        "palm_link": cfg.palm_link,
        "fingertip_links": cfg.fingertip_links,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
