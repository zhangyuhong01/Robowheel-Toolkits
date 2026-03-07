# -*- coding: utf-8 -*-
"""
触觉可视化工具包 view_tactile_tool。

入口脚本：项目根目录下的 view_tactile_tool.py
用法：python view_tactile_tool.py [--frame N] [command]
  command: select_frames | test_transform | test_pressure | test_gaussian | test_open3d
  无 command 时打开选帧界面，选帧后按 ENTER 可自动运行 transform -> pressure -> gaussian -> open3d
"""
from . import config
from . import io_utils
from . import frame_selection
from . import transform
from . import pressure
from . import gaussian
from . import open3d_viz
from . import gui

__all__ = [
    "config",
    "io_utils",
    "frame_selection",
    "transform",
    "pressure",
    "gaussian",
    "open3d_viz",
    "gui",
]
