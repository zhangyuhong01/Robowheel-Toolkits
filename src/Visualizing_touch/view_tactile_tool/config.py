# -*- coding: utf-8 -*-
"""触觉可视化工具：路径与全局配置。"""
import os

# 项目根目录（view_tactile_tool 包所在目录的上一级）
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(p):
    """相对路径会相对项目根目录解析为绝对路径。"""
    if not p or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(PACKAGE_ROOT, p))


# ==================== 运行前只需配置以下三项 ====================
DEFAULT_HDF5_PATH = _resolve_path(
    os.path.join("data", "hdf5", "100017", "episode_1963_204645_78_110104_merged.hdf5")
)
DEFAULT_OBJ_PATH = _resolve_path(
    os.path.join("data", "obj", "obj_100017", "baishikele.obj")
)
MANO_ASSETS_ROOT = _resolve_path(os.path.join("mano_v1_2"))

# 派生配置
DEFAULT_MANO_HDF5_PATH = DEFAULT_HDF5_PATH
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "output")

# 未在 select_frames 选择时使用的默认手（left / right）
DEFAULT_WHICH_HAND = "left"
# 可视化模式：只画手 / 只画物体 / 手+物体同时（仅对 test_open3d 生效）
VISUALIZATION_MODE = "both"  # "hand" | "object" | "both"
