# Robowheel :  A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning

[Yuhong Zhang](https://scholar.google.com/citations?user=oV7sxpYAAAAJ&hl=zh-CN)💡, Zihan Gao💡, Shengpeng Li, [Ling-Hao Chen](https://lhchen.top/), Kaisheng Liu,  Runqing Cheng, Xiao Lin, Junjia Liu, Zhuoheng Li, Jingyi Feng, Zheyan Huang, Jintian Lin, Zheyan Huang, Zhifang Liu, Haoqian Wang🌟

💡Equal Contribution, 🌟Corresponding Author  

📄 **[arXiv Paper](https://arxiv.org/abs/2512.02729)**,  🔗  **[Project Page](https://zhangyuhong01.github.io/Robowheel)**  🤗  **[HORA Dataset](https://huggingface.co/datasets/HORA-DB/HORA)**




## 📝 Abstract

![teaser](./assets/teaser.png)

We introduce RoboWheel, a data engine that converts hand–object interaction (HOI) videos into training ready supervision for cross-morphology robotic learning. From monocular RGB/RGB-D inputs, we perform high precision HOI reconstruction and enforce physical plausibility via a reinforcement learning (RL) optimizer that refines hand–object relative poses under contact and penetra tion constraints. The reconstructed, contact-rich trajectories are then retargeted to cross-embodiments, robot arms with simple end-effectors, dexterous hands, and humanoids, yielding executable actions and rollouts. To scale coverage, we build a simulation-augmented framework on Isaac Sim, with diverse domain randomization (embodiments, trajectories, object retrieval, background textures, hand motion mirroring), which enriches the distributions of trajectories and observations while preserving spatial relationships and physical plausibility. The entire data pipeline forms an end-to-end pipeline from video → reconstruction → retargeting → augmentation → data acquisition. To our knowledge, this provides the first quantitative evidence that HOI modalities can serve as effective super vision for robotic learning. Compared with teleoperation, RoboWheel is lightweight: a single monocular RGB(D) camera is sufficient to extract a universal, embodiment agnostic motion representation that could be flexibly retargeted across embodiments. We further assemble a large scale multimodal dataset combining multi-camera captures, monocular videos, and public HOI corpora for training and evaluating embodied models.



## 📌 To-Do List

Planned or ongoing work items:

- [x] Release part of HORA.
      
- [x] HORA replay in Issac lab 

- [ ] Release training code for our baseline models.

  - [x] Update guide for training Pi0(2026-01-14).

  - [ ] Release pretrained Pi0 model
      
  - [ ] Release guides for other models 

- [ ] Release the full set of HORA.



## 🧩 Training models with our data

### Pi0 


```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
```


We provide a script to convert HORA data to the LeRobot format. Download [convert_hdf5_to_lerobot_demo](https://github.com/zhangyuhong01/Robowheel-Toolkits/blob/main/scripts/convert_hdf5_to_lerobot_demo.py) and place it under openpi/blob/main/examples/libero.


```python
cd path_to_openpi
uv run examples/libero/convert_hdf5_to_lerobot_demo.py --hdf5_dir path_to_your_hdf5file --push_to_hub
```

Change your config in `openpi/src/openpi/training/config.py`, for example Pi0 model for LoRA fine-tuning:

```python
TrainConfig(
    # TODO
    name="NAME of YOUR TASK",
    # Here is an example of loading a pi0 model for LoRA fine-tuning.
    model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    # TODO
    data=LeRobotLiberoDataConfig(
        repo_id="YOUR_REPO_NAME",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.
    # TODO
    CheckpointWeightLoader("pathtocheckpoint/pi0_base/params"),
    num_train_steps=30_000,
    # The freeze filter defines which parameters should be frozen during training.
    # We have a convenience function in the model config that returns the default freeze filter
    # for the given model config for LoRA finetuning. Just make sure it matches the model config
    # you chose above.
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    # Turn off EMA for LoRA finetuning.
    ema_decay=None,
),
```
```python
uv run scripts/compute_norm_stats.py --config-name YOUR_CONFIG_NAME
uv run scripts/train.py YOUR_CONFIG_NAME --exp-name=YOUR_CONFIG_NAME --overwrite
```

## 👀 Using HORA in Issac Lab

We provide a toolkit to replay and visualize the HORA dataset using Isaac Sim.

### 1. Prerequisites
Ensure you have **Isaac Lab** installed. Please follow the official [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up the environment.

### 2. Install the Toolkit
Once the simulation environment is ready, install this package in editable mode:

```bash
pip install -e .

```
### 3. Run Visualization
You can visualize the data by running the replay script below. This command will automatically fetch a chunk of data from the HORA dataset for visualization:
```bash
python scripts/isaacsim_hora_replay.py

```


## 📖 Citation

```
@article{zhang2025robowheel,
  title={RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning},
  author={Zhang, Yuhong and Gao, Zihan and Li, Shengpeng and Chen, Ling-Hao and Liu, Kaisheng and Cheng, Runqing and Lin, Xiao and Liu, Junjia and Li, Zhuoheng and Feng, Jingyi and others},
  journal={arXiv preprint arXiv:2512.02729},
  year={2025}
}
```
