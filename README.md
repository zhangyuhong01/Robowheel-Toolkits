# Robowheel :  A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning

[Yuhong Zhang](https://scholar.google.com/citations?user=oV7sxpYAAAAJ&hl=zh-CN)💡, Zihan Gao💡, Shengpeng Li, [Ling-Hao Chen](https://lhchen.top/), Kaisheng Liu,  Runqing Cheng, Xiao Lin, Junjia Liu, Zhuoheng Li, Jingyi Feng, Zheyan Huang, Jintian Lin, Zheyan Huang, Zhifang Liu, Haoqian Wang🌟

💡Equal Contribution, 🌟Corresponding Author  

📄 **[arXiv Paper](https://arxiv.org/abs/2512.02729)**,  🔗  **[Project Page](https://zhangyuhong01.github.io/Robowheel)**




## 📝 Abstract

![teaser](./assets/teaser.png)

We introduce RoboWheel, a data engine that converts hand–object interaction (HOI) videos into training ready supervision for cross-morphology robotic learning. From monocular RGB/RGB-D inputs, we perform high precision HOI reconstruction and enforce physical plausibility via a reinforcement learning (RL) optimizer that refines hand–object relative poses under contact and penetra tion constraints. The reconstructed, contact-rich trajectories are then retargeted to cross-embodiments, robot arms with simple end-effectors, dexterous hands, and humanoids, yielding executable actions and rollouts. To scale coverage, we build a simulation-augmented framework on Isaac Sim, with diverse domain randomization (embodiments, trajectories, object retrieval, background textures, hand motion mirroring), which enriches the distributions of trajectories and observations while preserving spatial relationships and physical plausibility. The entire data pipeline forms an end-to-end pipeline from video → reconstruction → retargeting → augmentation → data acquisition. To our knowledge, this provides the first quantitative evidence that HOI modalities can serve as effective super vision for robotic learning. Compared with teleoperation, RoboWheel is lightweight: a single monocular RGB(D) camera is sufficient to extract a universal, embodiment agnostic motion representation that could be flexibly retargeted across embodiments. We further assemble a large scale multimodal dataset combining multi-camera captures, monocular videos, and public HOI corpora for training and evaluating embodied models.

## 📝 Train
### 📝 Train Pi0





## 📌 To-Do List

Planned or ongoing work items:

- [ ] Release part of HORA.

- [ ] Release training code for our baseline models.

- [ ] Release the full set of HORA.

  

