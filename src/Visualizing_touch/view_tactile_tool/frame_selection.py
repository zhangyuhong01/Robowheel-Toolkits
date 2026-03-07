# -*- coding: utf-8 -*-
"""触觉可视化工具：选帧与选帧 GUI。"""
import io
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from datetime import datetime

from read_tactile import DataLoaderV3

from . import config
from . import io_utils


def _sanitize_text_for_display(text):
    """去掉 emoji 与非常规字符，避免 matplotlib 显示时触发 Glyph missing 字体警告。"""
    if not text:
        return text
    result = []
    for c in text:
        if ord(c) <= 127 and (ord(c) >= 32 or c in "\n\t"):
            result.append(c)
        else:
            result.append(" ")
    return "".join(result)


def _show_no_valid_frames_dialog(start_frame, end_frame):
    """在界面弹窗提示：所选时间范围内没有有效触觉帧。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    msg = (
        "No valid tactile frames in selected range.\n\n"
        f"Range: Frame {start_frame} - {end_frame}\n\n"
        "Please select another time range or check your data.\n"
        "Close this window or click OK to continue."
    )
    ax.text(
        0.5, 0.65, msg, transform=ax.transAxes, fontsize=12,
        verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace", multialignment="center",
    )

    def on_ok(event):
        plt.close(fig)

    ax_btn = fig.add_axes([0.4, 0.2, 0.2, 0.1])
    btn = Button(ax_btn, "OK", color="lightblue")
    btn.on_clicked(on_ok)
    plt.show()


def _run_with_captured_stdout(func, *args, **kwargs):
    """运行函数并捕获 stdout 输出，返回 (返回值, 日志字符串)。同时保留控制台打印。"""
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
        def flush(self):
            for f in self.files:
                f.flush()

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, buf)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result, buf.getvalue()


def find_frames_with_tactile_data(
    hdf5_path, which_hand="right", quality_threshold=0.01,
    max_frames_to_check=None, frame_range=None
):
    """
    寻找有有效触觉数据的帧。
    Returns:
        dict: valid_frames, quality_scores, frame_stats
    """
    print("=== 开始分析触觉数据帧 ===")
    loader = DataLoaderV3(hdf5_path, which_hand=which_hand)
    total_frames = loader.num_frames

    if frame_range is not None:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        frame_pool = range(start_frame, end_frame + 1)
        print(f"限制检查范围: {start_frame} - {end_frame} ({len(frame_pool)} 帧)")
    else:
        frame_pool = range(total_frames)
        print(f"检查所有帧: {total_frames} 帧")

    if max_frames_to_check is None or len(frame_pool) <= max_frames_to_check:
        frames_to_check = list(frame_pool)
    else:
        step = max(1, len(frame_pool) // max_frames_to_check)
        frames_to_check = list(frame_pool)[::step]

    valid_frames = []
    quality_scores = []
    frame_stats = {}
    print(f"实际检查帧数: {len(frames_to_check)}")

    for i, frame_idx in enumerate(frames_to_check):
        if i % 20 == 0:
            print(f"进度: {i+1}/{len(frames_to_check)}")
        tactile_dict = loader.get_tactile_data_dict([frame_idx])
        total_non_zero = 0
        max_force = 0
        active_sensors = 0
        sensor_stats = {}
        for sensor_name, data in tactile_dict.items():
            sensor_data = data[frame_idx]
            non_zero_count = np.count_nonzero(sensor_data)
            max_val = np.max(np.abs(sensor_data))
            sensor_stats[sensor_name] = {
                "non_zero_count": non_zero_count,
                "max_value": max_val,
                "mean_value": np.mean(np.abs(sensor_data[sensor_data != 0])) if non_zero_count > 0 else 0,
            }
            total_non_zero += non_zero_count
            max_force = max(max_force, max_val)
            if non_zero_count > 0:
                active_sensors += 1
        quality_score = 0
        if total_non_zero > 0:
            quality_score += min(4, active_sensors * 0.5)
            data_density = total_non_zero / (len(tactile_dict) * 100)
            quality_score += min(3, data_density * 10)
            quality_score += min(3, max_force * 5)
        frame_stats[frame_idx] = {
            "total_non_zero": total_non_zero,
            "max_force": max_force,
            "active_sensors": active_sensors,
            "quality_score": quality_score,
            "sensor_stats": sensor_stats,
        }
        if quality_score > quality_threshold:
            valid_frames.append(frame_idx)
            quality_scores.append(quality_score)

    loader.close()
    if valid_frames:
        sorted_indices = np.argsort(quality_scores)[::-1]
        valid_frames = [valid_frames[i] for i in sorted_indices]
        quality_scores = [quality_scores[i] for i in sorted_indices]
    print(f"找到 {len(valid_frames)} 个有效触觉帧")
    return {"valid_frames": valid_frames, "quality_scores": quality_scores, "frame_stats": frame_stats}


def choose_hand_gui():
    """图形界面选择左手或右手。Returns: 'left' | 'right' | None"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.text(0.5, 0.8, "Choose Hand to Analyze", ha="center", va="center", fontsize=20, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.6, "Please select which hand contains the tactile data:", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    choice = {"hand": None}

    def on_left_click(event):
        choice["hand"] = "left"
        plt.close(fig)
    def on_right_click(event):
        choice["hand"] = "right"
        plt.close(fig)
    def on_cancel_click(event):
        choice["hand"] = None
        plt.close(fig)

    ax_left = plt.axes([0.2, 0.3, 0.2, 0.1])
    ax_right = plt.axes([0.6, 0.3, 0.2, 0.1])
    ax_cancel = plt.axes([0.4, 0.1, 0.2, 0.08])
    btn_left = Button(ax_left, "Left Hand", color="lightblue")
    btn_right = Button(ax_right, "Right Hand", color="lightgreen")
    btn_cancel = Button(ax_cancel, "Cancel", color="lightcoral")
    btn_left.on_clicked(on_left_click)
    btn_right.on_clicked(on_right_click)
    btn_cancel.on_clicked(on_cancel_click)
    plt.show()
    return choice["hand"]


def plot_tactile_overview(hdf5_path, which_hand):
    """Plot overview of tactile data intensity across all frames. Returns (frame_indices, tactile_intensities)."""
    print("Computing tactile data overview...")
    loader = DataLoaderV3(hdf5_path, which_hand=which_hand)
    total_frames = loader.num_frames
    frame_indices = []
    tactile_intensities = []
    step = max(1, total_frames // 500)
    for frame_idx in range(0, total_frames, step):
        try:
            tactile_dict = loader.get_tactile_data_dict([frame_idx])
            total_intensity = 0
            for sensor_name, data in tactile_dict.items():
                sensor_data = data[frame_idx]
                non_zero_data = sensor_data[sensor_data != 0]
                if len(non_zero_data) > 0:
                    total_intensity += np.sum(np.abs(non_zero_data))
            frame_indices.append(frame_idx)
            tactile_intensities.append(total_intensity)
            if frame_idx % (step * 20) == 0:
                print(f"Progress: {frame_idx}/{total_frames}")
        except Exception:
            continue
    loader.close()
    print(f"Computed overview for {len(frame_indices)} sample points")
    return np.array(frame_indices), np.array(tactile_intensities)


def interactive_range_selector(hdf5_path, which_hand):
    """Interactive range selector: 拖拽选择范围，点击「确认」或按 ENTER 确认；点击「取消」或按 ESC 取消。"""
    from matplotlib.widgets import RectangleSelector, Button

    frame_indices, tactile_intensities = plot_tactile_overview(hdf5_path, which_hand)
    if len(frame_indices) == 0:
        print("No tactile data found for overview")
        return None
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.18)
    ax.plot(frame_indices, tactile_intensities, "b-", linewidth=1, alpha=0.7)
    ax.fill_between(frame_indices, tactile_intensities, alpha=0.3)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Tactile Intensity")
    ax.set_title("Tactile Data Overview - Drag to select range, then click Confirm or press ENTER")
    ax.grid(True, alpha=0.3)
    selection_result = {"range": None, "confirmed": False}

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        selection_result["range"] = (int(x1), int(x2))
        ax.axvspan(x1, x2, alpha=0.3, color="red")
        ax.set_title(f"Selected range: Frame {int(x1)} - {int(x2)} (Click Confirm or press ENTER)")
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == "enter":
            selection_result["confirmed"] = True
            plt.close(fig)
        elif event.key == "escape":
            selection_result["range"] = None
            plt.close(fig)

    def on_confirm(event):
        selection_result["confirmed"] = True
        plt.close(fig)

    def on_cancel(event):
        selection_result["range"] = None
        plt.close(fig)

    rect_sel = RectangleSelector(
        ax, onselect, useblit=False, button=[1],
        minspanx=5, minspany=0, spancoords="data",
        interactive=False
    )
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    ax_confirm = fig.add_axes([0.35, 0.02, 0.12, 0.06])
    ax_cancel = fig.add_axes([0.53, 0.02, 0.12, 0.06])
    btn_confirm = Button(ax_confirm, "Confirm", color="lightgreen", hovercolor="green")
    btn_cancel = Button(ax_cancel, "Cancel", color="lightcoral", hovercolor="red")
    btn_confirm.on_clicked(on_confirm)
    btn_cancel.on_clicked(on_cancel)

    try:
        if hasattr(fig.canvas, "manager") and fig.canvas.manager.window:
            fig.canvas.manager.window.raise_()
            fig.canvas.manager.window.focus_force()
    except Exception:
        pass

    print("Drag on plot to select time range, then click [Confirm] or press ENTER. Click [Cancel] or ESC to cancel.")
    plt.show()
    if selection_result["confirmed"] and selection_result["range"]:
        start_frame, end_frame = selection_result["range"]
        print(f"Selected range: Frame {start_frame} - {end_frame}")
        return start_frame, end_frame
    print("Range selection cancelled")
    return None


def save_selected_frames(
    selected_frames, hdf5_path, which_hand=None, output_file=None, quality_scores=None
):
    """保存选定帧的信息到 OUTPUT_DIR。"""
    io_utils.ensure_output_dir()
    base = (output_file or "selected_frames.txt").replace(".txt", "")
    txt_path = io_utils.out_path(f"{base}.txt")
    json_path = io_utils.out_path(f"{base}.json")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hdf5_file": hdf5_path,
        "selected_frames": selected_frames,
        "frame_count": len(selected_frames),
    }
    if which_hand is not None:
        metadata["which_hand"] = which_hand
    if quality_scores is not None:
        if isinstance(quality_scores, dict):
            metadata["quality_scores"] = [quality_scores.get(f) for f in selected_frames]
        else:
            metadata["quality_scores"] = list(quality_scores)[: len(selected_frames)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# 选定的触觉数据帧\n")
        f.write(f"# 生成时间: {metadata['timestamp']}\n")
        f.write(f"# 数据文件: {hdf5_path}\n")
        if which_hand is not None:
            f.write(f"# 左右手: {which_hand}\n")
        f.write(f"# 帧数量: {len(selected_frames)}\n\n")
        for frame_idx in selected_frames:
            f.write(f"{frame_idx}\n")
    print(f"选定帧信息已保存到 {config.OUTPUT_DIR}:")
    print(f"   JSON格式: {json_path}")
    print(f"   文本格式: {txt_path}")


def visual_frame_selector(
    hdf5_path, which_hand=None, max_candidates=50,
    run_pipeline_on_confirm=False, obj_path=None
):
    """
    两阶段选帧：1) 范围选择 2) 详细选帧。
    Returns: (selected_frames, which_hand)
    """
    if which_hand is None:
        which_hand = choose_hand_gui()
        if which_hand is None:
            print("Hand selection cancelled")
            return [], None
    print(f"Starting Two-Stage Frame Selector for {which_hand} hand")
    print("\n=== Stage 1: Select Time Range ===")
    range_result = interactive_range_selector(hdf5_path, which_hand)
    if range_result is None:
        print("Range selection cancelled")
        return [], which_hand
    start_frame, end_frame = range_result
    print(f"Selected range: {start_frame} - {end_frame} ({end_frame - start_frame + 1} frames)")
    print("\n=== Stage 2: Analyze Frames in Selected Range ===")
    frame_analysis = find_frames_with_tactile_data(
        hdf5_path, which_hand,
        quality_threshold=0.001,
        max_frames_to_check=None,
        frame_range=(start_frame, end_frame),
    )
    candidates = frame_analysis["valid_frames"][:max_candidates]
    if not candidates:
        _show_no_valid_frames_dialog(start_frame, end_frame)
        return [], which_hand
    print(f"Found {len(candidates)} candidate frames, launching visualization interface...")
    if run_pipeline_on_confirm and not obj_path:
        obj_path = config.DEFAULT_OBJ_PATH
    frame_quality_dict = (
        dict(zip(frame_analysis["valid_frames"], frame_analysis["quality_scores"]))
        if frame_analysis.get("valid_frames") and frame_analysis.get("quality_scores")
        else {}
    )

    class FrameSelectorGUI:
        def __init__(
            self, hdf5_path, candidates, which_hand, obj_path=None,
            run_pipeline_on_confirm=False, frame_quality_dict=None
        ):
            self.hdf5_path = hdf5_path
            self.candidates = candidates
            self.which_hand = which_hand
            self.obj_path = obj_path or config.DEFAULT_OBJ_PATH
            self.run_pipeline_on_confirm = run_pipeline_on_confirm
            self.frame_quality_dict = frame_quality_dict or {}
            self.current_idx = 0
            self.selected_frames = []
            self.loader = DataLoaderV3(hdf5_path, which_hand=which_hand)
            self._bar_ax = None
            title = (
                "Tactile Data Frame Selector - ENTER run pipeline (or confirm), ESC exit"
                if run_pipeline_on_confirm
                else "Tactile Data Frame Selector - ENTER confirm, ESC exit"
            )
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle(title, fontsize=14)
            print("Preloading real tactile data...")
            self.frame_data = {}
            self.global_vmin = float("inf")
            self.global_vmax = float("-inf")
            for frame_idx in candidates:
                tactile_dict = self.loader.get_tactile_data_dict([frame_idx])
                self.frame_data[frame_idx] = tactile_dict
                for sensor_name, data in tactile_dict.items():
                    sensor_data = data[frame_idx]
                    if len(sensor_data) > 0:
                        self.global_vmin = min(self.global_vmin, np.min(sensor_data))
                        self.global_vmax = max(self.global_vmax, np.max(sensor_data))
            print(f"Loaded {len(candidates)} frames with real tactile data")
            print(f"Global data range: {self.global_vmin:.4f} - {self.global_vmax:.4f}")
            self.setup_gui()

        def setup_gui(self):
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
            self.update_display()

        def update_display(self):
            if getattr(self, "_bar_ax", None) is not None:
                self._bar_ax.remove()
                self._bar_ax = None
            frame_idx = self.candidates[self.current_idx]
            tactile_dict = self.frame_data[frame_idx]
            for ax in self.axes.flat:
                ax.clear()
            ax1 = self.axes[0, 0]
            sensor_names = list(tactile_dict.keys())
            sensor_activities = []
            for sensor_name in sensor_names:
                sensor_data = tactile_dict[sensor_name][frame_idx]
                activity = np.count_nonzero(sensor_data) / len(sensor_data)
                sensor_activities.append(activity)
            ax1.barh(range(len(sensor_names)), sensor_activities, color=plt.cm.viridis(np.array(sensor_activities)))
            ax1.set_yticks(range(len(sensor_names)))
            ax1.set_yticklabels(sensor_names, fontsize=8)
            ax1.set_xlabel("Activation Ratio")
            ax1.set_title(f"Sensor Activation Status (Frame {frame_idx})")
            ax1.grid(True, alpha=0.3)
            ax2 = self.axes[0, 1]
            all_forces = []
            for sensor_name, data in tactile_dict.items():
                sensor_data = data[frame_idx]
                non_zero_data = sensor_data[sensor_data != 0]
                if len(non_zero_data) > 0:
                    all_forces.extend(non_zero_data)
            if all_forces:
                ax2.hist(all_forces, bins=30, alpha=0.7, color="orange", edgecolor="black")
                ax2.set_xlabel("Force Value")
                ax2.set_ylabel("Frequency")
                ax2.set_title(f"Force Distribution (Total points: {len(all_forces)})")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No Valid Tactile Data", ha="center", va="center", transform=ax2.transAxes, fontsize=12)
                ax2.set_title("Force Distribution (No Data)")
            ax3 = self.axes[1, 0]
            if all_forces:
                max_len = max(len(tactile_dict[name][0]) for name in sensor_names)
                data_matrix = np.zeros((len(sensor_names), min(max_len, 100)))
                for i, sensor_name in enumerate(sensor_names):
                    sensor_data = tactile_dict[sensor_name][frame_idx]
                    data_len = min(len(sensor_data), 100)
                    data_matrix[i, :data_len] = sensor_data[:data_len]
                ax3.imshow(data_matrix, aspect="auto", cmap="hot", interpolation="nearest", vmin=self.global_vmin, vmax=self.global_vmax)
                ax3.set_yticks(range(len(sensor_names)))
                ax3.set_yticklabels(sensor_names, fontsize=8)
                ax3.set_xlabel("Data Point Index")
                ax3.set_title(f"Sensor Data Heatmap (Global Range: {self.global_vmin:.3f} - {self.global_vmax:.3f})")
            else:
                ax3.text(0.5, 0.5, "No Data to Display", ha="center", va="center", transform=ax3.transAxes, fontsize=12)
                ax3.set_title("Sensor Data Heatmap")
            ax4 = self.axes[1, 1]
            ax4.axis("off")
            info_text = f"""
Frame Info:
Current Frame: {frame_idx} ({self.current_idx + 1}/{len(self.candidates)})
Active Sensors: {sum(1 for a in sensor_activities if a > 0)}/{len(sensor_names)}
Total Data Points: {len(all_forces)}
Max Force Value: {max(all_forces) if all_forces else 0:.4f}

Selected Frames: {self.selected_frames}

Controls:
<- -> : Browse frames
SPACE: Select/deselect current frame
ENTER: Confirm selection and exit
ESC: Cancel and exit
"""
            if self.run_pipeline_on_confirm:
                info_text += "\n(ENTER = run pipeline for selected frame(s), then Open3D)"
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace")
            if frame_idx in self.selected_frames:
                ax4.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax4.transAxes, facecolor="green", alpha=0.2))
                ax4.text(0.5, 0.02, "[SELECTED]", transform=ax4.transAxes, ha="center", fontsize=12, color="green", weight="bold")
            self.fig.canvas.draw()

        def update_pipeline_display(self, step_name, log_text, steps_done=None, sub_progress=None):
            for ax in self.axes.flat:
                ax.clear()
                ax.axis("off")
            n_steps = 4
            if steps_done is None:
                steps_done = []
            running_idx = None
            if step_name:
                sn = step_name.lower()
                if "coordinate_transformation" in sn or "test_coordinate" in sn:
                    running_idx = 0
                elif "pressure_calculation" in sn or "test_pressure" in sn:
                    running_idx = 1
                elif "gaussian" in sn:
                    running_idx = 2
                elif "open3d" in sn:
                    running_idx = 3
            completed = len(steps_done)
            if sub_progress is None:
                sub_progress = 0.5 if running_idx is not None else 0.0
            sub_progress = max(0, min(1, sub_progress))
            pct = (completed + sub_progress) / n_steps * 100
            pct_int = int(round(pct))
            current_step = (running_idx if running_idx is not None else completed) + 1
            if not hasattr(self, "_bar_ax") or self._bar_ax is None:
                self._bar_ax = self.fig.add_axes([0.12, 0.38, 0.76, 0.28])
            self._bar_ax.clear()
            self._bar_ax.axis("off")
            bar_y, bar_h = 0.35, 0.5
            bar_x, bar_w = 0.05, 0.9
            self._bar_ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_w, bar_h, transform=self._bar_ax.transAxes, facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=1))
            fill_w = bar_w * pct_int / 100
            if fill_w > 0.002:
                self._bar_ax.add_patch(patches.Rectangle((bar_x, bar_y), fill_w, bar_h, transform=self._bar_ax.transAxes, facecolor="#3498db", edgecolor="none"))
            self._bar_ax.text(0.5, bar_y + bar_h / 2, f"{pct_int}%", transform=self._bar_ax.transAxes, fontsize=28, ha="center", va="center", fontfamily="monospace")
            self._bar_ax.text(0.5, bar_y - 0.15, f"Step {current_step}/{n_steps}", transform=self._bar_ax.transAxes, fontsize=11, ha="center", va="top", fontfamily="monospace")
            self._bar_ax.text(0.5, 0.02, _sanitize_text_for_display(step_name or "Idle"), transform=self._bar_ax.transAxes, fontsize=10, ha="center", va="bottom", fontfamily="monospace")
            self.fig.canvas.draw()
            plt.pause(0.02)

        def on_key_press(self, event):
            if event.key in ("right", "down"):
                self.current_idx = (self.current_idx + 1) % len(self.candidates)
                self.update_display()
            elif event.key in ("left", "up"):
                self.current_idx = (self.current_idx - 1) % len(self.candidates)
                self.update_display()
            elif event.key == " ":
                frame_idx = self.candidates[self.current_idx]
                if frame_idx in self.selected_frames:
                    self.selected_frames.remove(frame_idx)
                    print(f"Deselected frame {frame_idx}")
                else:
                    self.selected_frames.append(frame_idx)
                    print(f"Selected frame {frame_idx}")
                self.update_display()
            elif event.key == "enter":
                if self.run_pipeline_on_confirm and self.selected_frames:
                    quality_scores = {f: self.frame_quality_dict[f] for f in self.selected_frames if f in self.frame_quality_dict}
                    save_selected_frames(self.selected_frames, self.hdf5_path, which_hand=self.which_hand, quality_scores=quality_scores if quality_scores else None)
                    which_hand = self.which_hand
                    obj_path = self.obj_path
                    from . import transform as transform_mod
                    from . import pressure as pressure_mod
                    from . import gaussian as gaussian_mod
                    from . import open3d_viz as open3d_mod
                    self.update_pipeline_display("Running: test_coordinate_transformation...", "", steps_done=[])
                    def _cb1(c, t):
                        self.update_pipeline_display("Running: test_coordinate_transformation...", "", steps_done=[], sub_progress=c / t if t else 0)
                    transform_mod.test_coordinate_transformation(self.hdf5_path, which_hand=which_hand, mano_hdf5_path=None, progress_callback=_cb1)
                    self.update_pipeline_display("test_coordinate_transformation done", "", steps_done=[0])
                    plt.pause(0.05)
                    self.update_pipeline_display("Running: test_pressure_calculation...", "", steps_done=[0])
                    def _cb2(c, t):
                        self.update_pipeline_display("Running: test_pressure_calculation...", "", steps_done=[0], sub_progress=c / t if t else 0)
                    pressure_mod.test_pressure_calculation(obj_path, progress_callback=_cb2)
                    self.update_pipeline_display("test_pressure_calculation done", "", steps_done=[0, 1])
                    plt.pause(0.05)
                    self.update_pipeline_display("Running: test_gaussian_pressure_distribution...", "", steps_done=[0, 1])
                    def _cb3(c, t):
                        self.update_pipeline_display("Running: test_gaussian_pressure_distribution...", "", steps_done=[0, 1], sub_progress=c / t if t else 0)
                    gaussian_mod.test_gaussian_pressure_distribution(obj_path, progress_callback=_cb3)
                    self.update_pipeline_display("test_gaussian_pressure_distribution done", "", steps_done=[0, 1, 2])
                    plt.pause(0.05)
                    self.update_pipeline_display("Opening Open3D (close its window to return here)...", "", steps_done=[0, 1, 2])
                    self.fig.canvas.draw()
                    plt.pause(0.05)
                    open3d_mod.test_open3d_visualization(obj_path, frame_idx=None)
                    io_utils.clear_intermediate_files(self.selected_frames)
                    self.update_display()
                elif self.run_pipeline_on_confirm and not self.selected_frames:
                    print("Select at least one frame (SPACE) then press ENTER to run pipeline.")
                else:
                    print(f"Confirmed selection: {self.selected_frames}")
                    plt.close(self.fig)
            elif event.key == "escape":
                print("Selection cancelled")
                self.selected_frames = []
                plt.close(self.fig)

        def show(self):
            plt.show()
            self.loader.close()
            return self.selected_frames, self.which_hand

    gui = FrameSelectorGUI(hdf5_path, candidates, which_hand, obj_path=obj_path, run_pipeline_on_confirm=run_pipeline_on_confirm, frame_quality_dict=frame_quality_dict)
    selected_frames, which_hand = gui.show()
    return selected_frames, which_hand
