# SpaceRS-M3

项目简介：

面向太空智算的轻量化多模态遥感大模型应用探索。

当前技术路线：

- Qwen3-VL-2B-Instruct
- VRSBench
- 遥感图像描述
- 视觉问答
- 目标定位
- 后续开展 LoRA 微调
- 后续开展 INT8/INT4 量化
- 后续验证 SlimSAM 可插拔分割模块

成员分工：

- 梁瀚升：组长，主模型、训练、量化、系统集成和技术报告
- 胡杨：VRSBench 数据、评测和 SlimSAM 验证
- 朱福康：工程复现、资源统计、实验记录和提交封装

第一阶段目标：

- 跑通 Qwen3-VL-2B 单图推理
- 整理 30 条 VRSBench 联调样本
- 记录推理时间和峰值显存
- 建立统一 Git 协作流程

Windows 工作区路径：

```text
\\wsl.localhost\Ubuntu-20.04\home\lhs\spacers_m3
```

Linux 项目路径：

```text
/home/lhs/spacers_m3
```

打开目录的方式：

在 Windows 版 VS Code 中通过 File -> Open Folder 打开：

```text
\\wsl.localhost\Ubuntu-20.04\home\lhs\spacers_m3
```

不要要求用户使用 Remote WSL。
