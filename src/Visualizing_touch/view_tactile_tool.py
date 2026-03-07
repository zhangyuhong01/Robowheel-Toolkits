#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
触觉可视化工具入口脚本 view_tactile_tool。

将原 proj_point_to_obj.py 与 read_tactile.py 相关逻辑拆分为 view_tactile_tool 包，
本脚本为统一入口，用法与 proj_point_to_obj 一致。

用法:
  python view_tactile_tool.py                    # 选帧界面，选帧后 ENTER 可运行完整 pipeline
  python view_tactile_tool.py select_frames     # 可视化选帧
  python view_tactile_tool.py test_transform    # 坐标系变换
  python view_tactile_tool.py test_pressure     # 压力计算
  python view_tactile_tool.py test_gaussian     # 高斯压力分布
  python view_tactile_tool.py test_open3d       # Open3D 压力可视化
  python view_tactile_tool.py --frame N test_open3d  # 指定帧号可视化

也可: python -m view_tactile_tool [command]
路径与默认手等配置见 view_tactile_tool/config.py。
"""
# 使用交互式后端，避免 Windows 下按钮点击无反应（须在 import pyplot 之前）
import matplotlib
matplotlib.use("TkAgg")

from view_tactile_tool.__main__ import main

if __name__ == "__main__":
    main()
