# -*- coding: utf-8 -*-
"""触觉可视化工具：日志与打印。"""
from datetime import datetime


def print_header(title, level=1):
    """打印标题头部。"""
    if level == 1:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    elif level == 2:
        print(f"\n{'-'*60}")
        print(f"  {title}")
        print(f"{'-'*60}")
    else:
        print(f"\n  {title}")


def print_status(message, status="INFO"):
    """打印状态信息。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "INFO":
        print(f"[{timestamp}] INFO  | {message}")
    elif status == "SUCCESS":
        print(f"[{timestamp}] OK    | {message}")
    elif status == "ERROR":
        print(f"[{timestamp}] ERROR | {message}")
    elif status == "WARN":
        print(f"[{timestamp}] WARN  | {message}")


def print_data(label, value, unit=""):
    """打印数据信息。"""
    print(f"          DATA  | {label}: {value} {unit}")


def print_progress(step, total, description=""):
    """打印进度信息。"""
    percentage = (step / total) * 100 if total else 0
    print(f"          PROG  | Step {step}/{total} ({percentage:.1f}%) - {description}")


def print_file_output(filename, description=""):
    """打印文件输出信息。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] FILE  | Generated: {filename} {description}")


def print_separator():
    """打印分隔线。"""
    print(f"          {'─'*50}")
