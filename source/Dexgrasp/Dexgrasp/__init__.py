# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments when Isaac Lab is available.  Pure utility modules
# such as Dexgrasp.utils.demograsp_yaml should still import in a plain Python
# shell for static migration checks.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise

# Register UI extensions when Omniverse modules are available.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if exc.name not in {"omni", "isaacsim"}:
        raise
