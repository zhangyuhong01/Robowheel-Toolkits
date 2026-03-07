# -*- coding: utf-8 -*-
"""触觉可视化工具：主任务选择 GUI（Tkinter）。"""


def main_task_launcher_gui():
    """
    主界面：用 Tkinter 按钮选择任务（在 Windows 上比 matplotlib 按钮更稳定）。
    Returns:
        str: 子命令 'select_frames' | 'test_transform' | 'test_pressure' | 'test_gaussian' | 'test_open3d'，
             或 None（用户点击退出）
    """
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        tk = None

    if tk is None:
        print("Tkinter not available. Usage: python view_tactile_tool.py <command>")
        print("  select_frames | test_transform | test_pressure | test_gaussian | test_open3d")
        return None

    chosen = [None]

    root = tk.Tk()
    root.title("Tactile Projection & Pressure Visualization")
    root.resizable(True, True)
    root.minsize(420, 380)

    title_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
    label_font = tkfont.Font(family="Segoe UI", size=10)
    btn_font = tkfont.Font(family="Segoe UI", size=10)

    frame = tk.Frame(root, padx=24, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="Tactile Projection & Pressure Visualization", font=title_font).pack(pady=(0, 8))
    tk.Label(frame, text="Please select a task to run:", font=label_font).pack(pady=(0, 16))

    def on_click(cmd):
        chosen[0] = cmd
        root.quit()
        root.destroy()

    tasks = [
        ("1. Select Frames", "select_frames"),
        ("2. Coord Transform", "test_transform"),
        ("3. Pressure Calc", "test_pressure"),
        ("4. Gaussian Pressure", "test_gaussian"),
        ("5. Open3D Visualize", "test_open3d"),
    ]
    for label, cmd in tasks:
        btn = tk.Button(frame, text=label, font=btn_font, width=28, command=lambda c=cmd: on_click(c))
        btn.pack(pady=6)

    tk.Button(frame, text="Exit", font=btn_font, width=28, command=lambda: on_click(None)).pack(pady=16)

    root.mainloop()
    return chosen[0]
