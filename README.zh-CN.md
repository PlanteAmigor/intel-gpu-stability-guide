# Intel GPU AI 训练与推理问题记录

[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2026.04%20LTS-E95420)](https://ubuntu.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+xpu-EE4C2C)](https://pytorch.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.1-00A3E0)](https://docs.openvino.ai/)

本目录收录了在特定 Intel 平台（Arrow Lake + Arc GPU + NPU）上运行 AI 训练和推理时遇到的问题记录与缓解方案。

> ⚠️ **免责声明：** 本文档所述的观察均基于特定硬件/软件/模型组合，**可能不适用于其他环境**。详见各文档中的详细免责说明。

🌐 **语言:** [English](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/README.md) · [中文](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/README.zh-CN.md)

---

## 硬件环境

| 部件 | 型号 |
|------|------|
| **CPU** | Intel Core Ultra 9 285H (Arrow Lake, 6P+8E, 14核) |
| **iGPU** | Intel Arc Graphics (8 Xe-core, 共享内存 128GB) |
| **dGPU** | Intel Arc Pro 130T/140T (Arrow Lake-P, PCI ID `8086:7d51`) |
| **NPU** | Intel AI Boost (`/dev/dri/renderD128`) |
| **内存** | 128 GB DDR5 (与 iGPU 共享) |

## 软件环境

| 软件 | 版本 |
|------|------|
| **OS** | Ubuntu 26.04 LTS (Linux Kernel 7.0) |
| **PyTorch** | 2.12.0+xpu |
| **OpenVINO** | 2026.1 |
| **Python** | 3.14 |
| **oneAPI** | 2026.0 (IntelLLVM, MKL, DNNL, TBB) |
| **GPU 驱动** | `libze-intel-gpu1 26.14.37833.4` |

---

## 文档索引

### 1. [独立显卡 (dGPU) 稳定性指南](intel_gpu_aitrain001.md)

**适用场景：** OpenVINO + Qwen3 系列模型在 Intel Arc 独立显卡上的推理负载。

**核心发现（本次测试中）：**
- 持续 GPU 占用率 >90% → Kernel Panic / 段错误
- NaN 输出 → 驱动崩溃前兆（非量化精度问题）
- 缓解：小 batch + 频繁冷却间隔 + INT8 量化

| 文件 | 内容 |
|------|------|
| [intel_gpu_aitrain001.md](intel_gpu_aitrain001.md) | 完整问题分析（中英双语） |

---

### 2. [集成显卡 (iGPU) Transformer 反向传播崩溃报告](xpu_backward_issue.md)

**适用场景：** PyTorch XPU 后端在 Arc iGPU 上的 Transformer 训练。

**核心发现（本次测试中）：**
- `nn.TransformerEncoderLayer` / `F.scaled_dot_product_attention` 反向传播可能崩溃
- 错误类型：`RuntimeError`（负数维度/整数溢出）或 `IndexError`（索引越界）
- 即使 22M 参数小模型（hidden=512, heads=8, batch=4）也可能崩溃
- AMP BF16 可能升级为整机冻结（驱动崩溃）
- Gated CNN / Conv2d 等无注意力架构运行稳定

| 文件 | 内容 |
|------|------|
| [xpu_backward_issue.md](xpu_backward_issue.md) | 完整问题分析（中英双语） |

---

## 总结对比

| 场景 | GPU类型 | 框架 | 状态 | 推荐替代方案 |
|------|---------|------|------|------------|
| **Transformer 训练** | iGPU | PyTorch XPU | ❌ 反向传播崩溃 | 使用 Gated CNN / CPU 训练 |
| **CNN 训练** | iGPU | PyTorch XPU | ✅ 稳定 | — |
| **大模型推理 (Qwen3)** | dGPU | OpenVINO | ⚠️ 需冷却策略 | INT8 + 小 batch + 冷却 |
| **大模型推理** | NPU | OpenVINO | ✅ 已测试 | 需静态 shape |
| **大模型推理** | CPU | OpenVINO | ✅ 稳定 | 性能最慢 |

---

## 外部链接

- [Intel GPU 稳定性指南 (GitHub)](https://github.com/PlanteAmigor/intel-gpu-stability-guide) — 本仓库的外部镜像
- [PyTorch XPU 官方文档](https://docs.pytorch.org/docs/2.12/xpu.html)
- [OpenVINO 官方文档](https://docs.openvino.ai/)

---

*本文档记录的是特定环境下的实测结果，仅供参考。不同硬件/驱动/框架版本组合可能表现不同。*
