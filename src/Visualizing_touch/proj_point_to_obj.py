#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
触觉投影与压力可视化 - 兼容入口脚本。

逻辑已拆分为 view_tactile_tool 包，本文件仅作为兼容入口：
运行「python proj_point_to_obj.py [--frame N] [command]」等价于
「python view_tactile_tool.py [--frame N] [command]」。

子命令: select_frames | test_transform | test_pressure | test_gaussian | test_open3d
路径与默认手等配置见 view_tactile_tool/config.py。
"""
import matplotlib
matplotlib.use("TkAgg")

from view_tactile_tool.__main__ import main

if __name__ == "__main__":
    main()
