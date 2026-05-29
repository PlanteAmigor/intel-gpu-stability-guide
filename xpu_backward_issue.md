# Attention Backward Crash on Intel XPU: A Case Study

[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2026.04%20LTS-E95420)](https://ubuntu.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+xpu-EE4C2C)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-Arc%20iGPU%20(Meteor%20Lake)-00A3E0)](https://www.intel.com/)

> **Disclaimer:** The observations below were made on a specific hardware/software combination. They may not represent general Intel XPU behavior or apply to other configurations.

**A documented case of nn.TransformerEncoderLayer / F.scaled_dot_product_attention backward pass crashing on Intel Arc integrated GPU (Meteor Lake) with PyTorch XPU backend.**

---

## Observation Methodology

- **Workload:** Transformer training with nn.TransformerEncoderLayer (d_model=2048, nhead=16, batch_first=True), Gated CNN training, AlphaZero CNN training
- **Crash detection:** RuntimeError tracking, process exit code monitoring, kernel log inspection for GPU driver panics
- **Tested variants:** PyTorch XPU native, F.scaled_dot_product_attention replacement, AMP BF16, periodic memory cleanup

---

## Test Platform

| Category | Details |
|----------|---------|
| **CPU** | Intel Core Ultra 9 285H (Meteor Lake, 6P+8E, 14 cores) |
| **GPU** | Intel Arc iGPU (8 Xe-core, shared memory) |
| **RAM** | 128 GB DDR5 (shared with GPU) |
| **OS** | Ubuntu 26.04 LTS |
| **PyTorch** | 2.12.0+xpu |
| **Python** | 3.14 |

---

## The Problem

When training Transformer-based models (nn.TransformerEncoderLayer or F.scaled_dot_product_attention) on Intel XPU, the backward pass consistently crashes with memory allocation errors, segfaults, or system freezes.

**Minimal reproduction:**

```python
import torch, torch.nn as nn
m = nn.TransformerEncoderLayer(2048, 16, batch_first=True).to('xpu')
x = torch.randn(8, 512, 2048, device='xpu')
m(x).sum().backward()  # crash
```

**Error messages observed:**

```
RuntimeError: Trying to create tensor with negative dimension -79243236477491020: [-79243236477491020]
```

The numeric value varies per run (-7.9e16 to -5.0e17), suggesting integer overflow in memory allocation calculation.

```
IndexError: select(): index -1 out of range for tensor of size [0] at dimension 0
```

In severe cases (e.g., with AMP BF16 enabled), the entire system freezes requiring a hard reboot — the GPU driver itself crashes, not just the Python process.

---

## Works vs Does Not Work

| Operation | XPU Status |
|-----------|-----------|
| **nn.Conv2d** forward + backward | ✅ Stable |
| **nn.Linear** forward + backward | ✅ Stable |
| **nn.BatchNorm2d** forward + backward | ✅ Stable |
| **nn.TransformerEncoderLayer** forward | ✅ OK |
| **nn.TransformerEncoderLayer** backward | ❌ **Crashes** |
| **F.scaled_dot_product_attention** forward | ✅ OK |
| **F.scaled_dot_product_attention** backward | ❌ **Crashes** |
| CPU (device='cpu'), all operations | ✅ **Stable** |

---

## Key Findings

### 1. Attention Backward Is the Root Cause

CNN-only models (Conv2d, ResNet, Gated CNN) run stably on XPU for both training and inference. The crash is specific to the backward pass of attention-like operations (nn.MultiheadAttention, F.scaled_dot_product_attention). The forward pass of the same operations completes successfully.

### 2. Scale-Sensitive

Small models (hidden_size=512, batch=2) may run without crashing. The failure is nearly guaranteed with larger hidden sizes (2048) and batch sizes (8+). This suggests a memory allocation calculation that overflows at certain tensor size thresholds.

### 3. AMP BF16 Worsens the Problem

Enabling AMP BF16 (torch.amp.autocast) on XPU does not fix the issue — instead it escalates it from a recoverable Python exception to a full system freeze, requiring hard reboot. The GPU driver itself crashes.

### 4. Workarounds Have Limited Effect

- Periodic `torch.xpu.empty_cache()` + `gc.collect()` — only delays the crash
- `torch.xpu.synchronize()` — no effect
- Gated CNN (no attention) — completely avoids the issue

---

## Impact

This bug effectively prevents training any Transformer-based model on Intel XPU with PyTorch, including:
- BERT-style encoders
- GPT-style decoders
- Cross-attention layers
- Any model using nn.TransformerEncoderLayer / nn.TransformerDecoderLayer

CNN-based models (image classification, AlphaZero-style game AI, Gated CNN text generation) are unaffected.

---

## Summary

On the tested platform (Intel Core Ultra 9 285H + Arc iGPU + PyTorch 2.12.0+xpu), the backward pass of attention operations crashes reliably under moderate model sizes. The root cause appears to be in the Intel XPU backend's memory allocation for attention gradient computation — not in the model code, framework, or user configuration. A fix would likely need to come from Intel's oneAPI or PyTorch XPU backend team.

The practical workaround is to use attention-free architectures (CNN, Gated CNN, MLP-only) for training on this platform.

---

# Intel XPU 注意力反向传播崩溃问题记录

> **免责声明：** 以下观察基于特定硬件/软件组合，未必代表 Intel XPU 普遍行为。

**关于 nn.TransformerEncoderLayer / F.scaled_dot_product_attention 反向传播在 Intel Meteor Lake Arc iGPU + PyTorch XPU 后端上崩溃的个案记录。**

---

## 测试平台

| 类别 | 详情 |
|------|------|
| **CPU** | Intel Core Ultra 9 285H (Meteor Lake, 6P+8E, 14核) |
| **GPU** | Intel Arc iGPU (8 Xe-core, 共享内存) |
| **内存** | 128 GB DDR5（与 GPU 共享） |
| **系统** | Ubuntu 26.04 LTS |
| **PyTorch** | 2.12.0+xpu |
| **Python** | 3.14 |

---

## 问题描述

在 Intel XPU 上训练 Transformer 模型（nn.TransformerEncoderLayer 或 F.scaled_dot_product_attention）时，反向传播持续崩溃，表现为内存分配错误、段错误或系统冻结。

**最小复现代码：**

```python
import torch, torch.nn as nn
m = nn.TransformerEncoderLayer(2048, 16, batch_first=True).to('xpu')
x = torch.randn(8, 512, 2048, device='xpu')
m(x).sum().backward()  # 崩溃
```

**观察到的错误信息：**

```
RuntimeError: Trying to create tensor with negative dimension -79243236477491020: [-79243236477491020]
```

数值每次不同（在 -7.9e16 到 -5.0e17 之间），疑似内存分配计算中的整数溢出。

```
IndexError: select(): index -1 out of range for tensor of size [0] at dimension 0
```

严重时（如启用 AMP BF16）整机冻结，需硬重启——GPU 驱动本身崩溃，不只是 Python 进程挂掉。

---

## 能跑 vs 不能跑

| 操作 | XPU 状态 |
|------|---------|
| **nn.Conv2d** 前向+反向 | ✅ 稳定 |
| **nn.Linear** 前向+反向 | ✅ 稳定 |
| **nn.BatchNorm2d** 前向+反向 | ✅ 稳定 |
| **nn.TransformerEncoderLayer** 前向 | ✅ 正常 |
| **nn.TransformerEncoderLayer** 反向 | ❌ **崩溃** |
| **F.scaled_dot_product_attention** 前向 | ✅ 正常 |
| **F.scaled_dot_product_attention** 反向 | ❌ **崩溃** |
| CPU (device='cpu')，所有操作 | ✅ **稳定** |

---

## 关键发现

### 1. 注意力反向传播是根因

纯 CNN 模型（Conv2d、ResNet、Gated CNN）在 XPU 上运行稳定。崩溃仅发生在注意力类操作的反向传播中。同操作的前向传播正常完成。

### 2. 规模敏感性

小模型（hidden_size=512, batch=2）可能不崩溃。随着参数规模增大（hidden_size=2048, batch=8+）几乎必现，表明内存在分配计算在特定张量尺寸阈值时溢出。

### 3. AMP BF16 会加剧问题

开启 AMP BF16 不仅不修复问题，反而将其从可恢复的 Python 异常升级为整机冻结，需硬重启——GPU 驱动直接崩溃。

### 4. 缓解措施效果有限

- 定时 `torch.xpu.empty_cache()` + `gc.collect()`——仅推迟崩溃
- `torch.xpu.synchronize()`——无效果
- 改用 Gated CNN（无注意力）——完全规避问题

---

## 影响范围

此 bug 事实上阻止了在 Intel XPU 上使用 PyTorch 训练任何 Transformer 模型，包括：
- BERT 类编码器
- GPT 类解码器
- 交叉注意力层
- 任何使用 nn.TransformerEncoderLayer / nn.TransformerDecoderLayer 的模型

CNN 模型（图像分类、AlphaZero 游戏 AI、Gated CNN 文本生成）不受影响。

---

## 总结

在本次测试平台（Intel Core Ultra 9 285H + Arc iGPU + PyTorch 2.12.0+xpu）上，注意力操作的反向传播在中等模型规模下稳定崩溃。根因疑似在 Intel XPU 后端的注意力梯度计算内存分配中——并非模型代码、框架或用户配置的问题。修复需要来自 Intel oneAPI 或 PyTorch XPU 后端团队的更新。

当前可行的方案是使用无注意力的架构（CNN、Gated CNN、纯 MLP）在此平台上进行训练。
