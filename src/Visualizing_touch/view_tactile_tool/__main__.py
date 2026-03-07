# -*- coding: utf-8 -*-
"""触觉可视化工具包入口：python -m view_tactile_tool [--frame N] [command]"""
import sys
import argparse

from . import config
from . import frame_selection
from . import transform
from . import pressure
from . import gaussian
from . import open3d_viz


def main():
    parser = argparse.ArgumentParser(
        description="触觉投影与压力可视化：HDF5 触觉数据 + OBJ 模型（路径等见 view_tactile_tool/config.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例: python -m view_tactile_tool  (打开选帧界面)  |  "
            "python -m view_tactile_tool select_frames  |  "
            "python -m view_tactile_tool --frame 70 test_open3d"
        ),
    )
    parser.add_argument("--frame", type=int, default=None, help="可视化指定帧号（不指定则用 selected_frames 中第一个存在的）")
    parser.add_argument("command", nargs="?", default=None, help="子命令")
    args = parser.parse_args()
    hdf5_path = config.DEFAULT_HDF5_PATH
    obj_path = config.DEFAULT_OBJ_PATH
    frame_for_viz = args.frame
    cmd = args.command

    if cmd is None:
        selected_frames, which_hand = frame_selection.visual_frame_selector(
            hdf5_path, which_hand=None,
            run_pipeline_on_confirm=True, obj_path=obj_path
        )
        if selected_frames:
            frame_selection.save_selected_frames(selected_frames, hdf5_path, which_hand=which_hand)
        sys.exit(0)

    if cmd == "test_transform":
        print("Starting coordinate transformation test")
        which_hand = transform.get_which_hand_from_selected_frames() or config.DEFAULT_WHICH_HAND
        transform.test_coordinate_transformation(hdf5_path, which_hand=which_hand, mano_hdf5_path=None)

    elif cmd == "test_pressure":
        print("Starting pressure calculation test")
        pressure.test_pressure_calculation(obj_path)

    elif cmd == "test_gaussian":
        print("Starting Gaussian pressure distribution test")
        gaussian.test_gaussian_pressure_distribution(obj_path)

    elif cmd == "test_open3d":
        print("Starting Open3D pressure visualization test")
        open3d_viz.test_open3d_visualization(obj_path, frame_idx=frame_for_viz)

    elif cmd == "select_frames":
        print("Starting visual frame selection process")
        selected_frames, which_hand = frame_selection.visual_frame_selector(hdf5_path, which_hand=None)
        if selected_frames:
            frame_selection.save_selected_frames(selected_frames, hdf5_path, which_hand=which_hand)
            print(f"Saved {len(selected_frames)} selected frames (which_hand={which_hand})")
            print("\n" + "=" * 60)
            print("Task 0 Complete: Frame Selection Finished!")
            print("=" * 60)
            print(f"Selected frames: {selected_frames}")
            print(f"Frames saved to: {config.OUTPUT_DIR} (selected_frames.txt, selected_frames.json)")
            print("Next: run test_transform -> test_pressure -> test_gaussian -> test_open3d")
            print("=" * 60)
        else:
            print("No frames selected")
            print("Tip: Use <- -> to browse, SPACE to select, ENTER to confirm, ESC to cancel")

    else:
        print("Usage: python view_tactile_tool.py [--frame N] [command]")
        print("  python view_tactile_tool.py                 # 选帧界面")
        print("  python view_tactile_tool.py select_frames   # 可视化选帧")
        print("  python view_tactile_tool.py test_transform  # 坐标系变换")
        print("  python view_tactile_tool.py test_pressure   # 压力计算")
        print("  python view_tactile_tool.py test_gaussian   # 高斯压力分布")
        print("  python view_tactile_tool.py test_open3d     # Open3D 压力可视化")


if __name__ == "__main__":
    main()
