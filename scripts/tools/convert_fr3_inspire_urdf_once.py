"""Convert the step-1 FR3 + Inspire Tac URDF into a temporary USD asset."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB_ROOT = Path("/home/windsky/IsaacLab")
DEFAULT_INPUT = REPO_ROOT / "source" / "Dexgrasp" / "Dexgrasp" / "assets" / "robots" / "inspire_tac" / (
    "fr3_inspire_tac_L_right_safety.urdf"
)
DEFAULT_OUTPUT = REPO_ROOT / "source" / "Dexgrasp" / "Dexgrasp" / "assets" / "robots" / "_tmp_usd" / (
    "fr3_inspire_tac"
) / "fr3_inspire_tac.usd"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the DemoGrasp FR3+Inspire URDF to a temporary USD.")
    parser.add_argument("--isaaclab_root", type=Path, default=DEFAULT_ISAACLAB_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry_run", action="store_true", help="Print the conversion command without running it.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_script = args.isaaclab_root / "scripts" / "tools" / "convert_urdf.py"
    command = [
        str(args.isaaclab_root / "isaaclab.sh"),
        "-p",
        str(convert_script),
        str(args.input),
        str(args.output),
        "--fix-base",
        "--joint-stiffness",
        "80.0",
        "--joint-damping",
        "4.0",
        "--joint-target-type",
        "position",
        "--headless",
    ]
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

