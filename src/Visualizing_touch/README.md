# 触觉可视化工具 (Visualizing Touch)

基于 HDF5 触觉数据与 OBJ 物体模型的触觉投影与压力可视化工具：选帧、坐标系变换、压力计算、高斯压力分布与 Open3D 可视化。

---

## 一、环境准备（Conda）

### 1. 安装 Miniconda / Anaconda

若尚未安装 Conda，请先安装其一：

- **Miniconda**（推荐，体积小）：<https://docs.conda.io/en/latest/miniconda.html>
- **Anaconda**：<https://www.anaconda.com/download>

安装后重启终端，确认可用：

```bash
conda --version
```

### 2. 创建并激活环境

在项目根目录下执行：

```bash
# 进入项目目录
cd e:\data\pipeline\Visualizing_touch

# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate visualizing_touch
```

若已存在同名环境，可先删除再创建：

```bash
conda env remove -n visualizing_touch
conda env create -f environment.yml
conda activate visualizing_touch
```

### 3. 可选：手动安装依赖

若不用 `environment.yml`，可手动创建环境并安装包：

```bash
conda create -n visualizing_touch python=3.9 -y
conda activate visualizing_touch

# 核心依赖（建议用 conda 安装 PyTorch 与 Open3D）
conda install numpy scipy h5py opencv matplotlib pytorch open3d -c pytorch -c conda-forge -y
```

### 4. 可选：视频解码（ffmpeg）

若需要从 HDF5 中解码相机视频（如 `read_tactile` 中的 h265 解码），请安装 **ffmpeg** 并加入 PATH：

- Windows：<https://ffmpeg.org/download.html> 或 `winget install ffmpeg`
- 安装后终端执行 `ffmpeg -version` 确认可用。

---

## 二、配置与数据

- **默认路径**在 `view_tactile_tool/config.py` 中配置：
  - `DEFAULT_HDF5_PATH`：触觉 HDF5 文件（如 `data/hdf5/100017/episode_*.hdf5`）
  - `DEFAULT_OBJ_PATH`：物体 OBJ 模型（如 `data/obj/obj_100017/baishikele.obj`）
  - `MANO_ASSETS_ROOT`：MANO 资源目录（默认 `mano_v1_2`）
- 将你的 HDF5 与 OBJ 放到对应目录，或直接修改上述路径。

---

## 三、运行方式

在**已激活**的 conda 环境下，在项目根目录执行：

### 主入口（选帧 + 完整 pipeline）

```bash
# 打开选帧界面，选帧后按 ENTER 可运行完整 pipeline
python view_tactile_tool.py
# 或
python -m view_tactile_tool
```

### 子命令

| 命令 | 说明 |
|------|------|
| `python view_tactile_tool.py select_frames` | 仅做可视化选帧并保存 |
| `python view_tactile_tool.py test_transform` | 坐标系变换测试 |
| `python view_tactile_tool.py test_pressure` | 压力计算测试 |
| `python view_tactile_tool.py test_gaussian` | 高斯压力分布测试 |
| `python view_tactile_tool.py test_open3d` | Open3D 压力可视化 |
| `python view_tactile_tool.py --frame N test_open3d` | 指定帧号 N 的 Open3D 可视化 |

兼容入口（与上面等价）：

```bash
python proj_point_to_obj.py
python proj_point_to_obj.py select_frames
# ...
```

### 选帧操作说明

- **← / →**：切换帧  
- **空格**：选中/取消当前帧  
- **Enter**：确认并保存选中帧（可选继续跑 pipeline）  
- **ESC**：取消  

选帧结果会保存到 `output/selected_frames.txt` 和 `output/selected_frames.json`。

---

## 四、项目结构概览

```
Visualizing_touch/
├── environment.yml       # Conda 环境定义
├── README.md             # 本说明
├── view_tactile_tool.py  # 主入口
├── proj_point_to_obj.py  # 兼容入口
├── read_tactile.py       # HDF5 触觉数据加载与解码
├── view_tactile_tool/    # 触觉可视化包
│   ├── config.py         # 路径与全局配置
│   ├── frame_selection.py # 选帧界面与 pipeline 串联
│   ├── transform.py      # 坐标系变换
│   ├── pressure.py       # 压力计算
│   ├── gaussian.py       # 高斯压力分布
│   ├── open3d_viz.py     # Open3D 压力可视化
│   └── ...
├── data/
│   ├── hdf5/             # HDF5 触觉数据
│   ├── obj/              # OBJ 物体模型
│   └── video/            # 解码后的视频（可选）
├── output/               # 选帧结果与输出图
├── mano_v1_2/            # MANO 手部模型资源
├── manopth/              # MANO PyTorch 层（可选）
└── manotorch/            # MANO PyTorch 工具（可选）
```

---

## 五、常见问题

- **Windows 下选帧界面按钮无反应**：入口脚本已设置 `matplotlib.use("TkAgg")`，若仍有问题请确认已安装 Tk（Conda 默认带 Tk）。
- **找不到 HDF5 或 OBJ**：检查 `view_tactile_tool/config.py` 中的 `DEFAULT_HDF5_PATH`、`DEFAULT_OBJ_PATH` 是否指向实际文件。
- **视频解码报错**：确认已安装 ffmpeg 且在 PATH 中；若不需要解码视频可忽略。
- **MANO / 手部相关报错**：当前主流程针对 exoskeleton 手部数据；MANO 数据需额外配置，见 `data/hdf5/mano_tactile.md`。

---

## 六、依赖摘要

| 依赖 | 用途 |
|------|------|
| Python 3.9 | 运行环境 |
| numpy, scipy | 数值与几何计算 |
| h5py | HDF5 读写 |
| opencv | 图像/视频处理 |
| matplotlib | 选帧界面与绘图 |
| pytorch | 触觉数据处理（read_tactile 等） |
| open3d | 压力云图 3D 可视化 |
| ffmpeg（可选） | 相机视频解码 |

环境创建完成后，从「二、配置与数据」和「三、运行方式」按需修改路径并运行即可。
