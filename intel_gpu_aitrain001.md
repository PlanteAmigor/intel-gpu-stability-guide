# Intel Arc GPU Stability Guide for AI Workloads

[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2026.04%20LTS-E95420)](https://ubuntu.com/)
[![Kernel](https://img.shields.io/badge/Kernel-7.0.0--15--generic-blue)](https://kernel.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.1-00A3E0)](https://docs.openvino.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+xpu-EE4C2C)](https://pytorch.org/)

> ⚠️ **Disclaimer:** The observations below were made on a specific hardware/software combination running specific models (detailed below). They may **not** represent general Intel Arc GPU behavior or apply to other configurations. Treat these as case study notes, not universal conclusions.

**A practical guide to preventing GPU driver crashes (Kernel Panic, Segfault) on Intel Arc discrete GPUs — observed during sustained AI inference workloads with Qwen3-series models.**

---

## Observation Methodology

The data in this document comes from the following setup and measurement methods:

- **Workload:** Batch-encoding Chinese classical poetry text passages (~7000 poems, 100–500 chars each) using Qwen3-Embedding-4B via OpenVINO
- **Monitoring tool:** `intel_gpu_top` (part of `intel-gpu-tools`), sampled at 1000ms intervals
- **Monitoring duration:** 70-second continuous capture during active embedding inference
- **Crash detection:** Process exit code monitoring (exit 139 = segfault), kernel log (`dmesg`) for kernel panic events, and output tensor inspection for NaN/Inf values
- **Comparison scope:** Same model, same data, same GPU — only quantization format (INT4 vs INT8), batch size (20 vs 10), and cooling strategy were varied

---

## Platforms & Models Tested

| Category | Details |
|----------|---------|
| **GPU** | Intel Arc Pro 130T/140T (Arrow Lake-P, PCI ID `8086:7d51`) |
| **NPU** | Intel AI Boost (`/dev/dri/renderD128`) |
| **RAM** | 64 GB |
| **OS** | Ubuntu 26.04 LTS |
| **Kernel** | `7.0.0-15-generic` |
| **Frameworks** | OpenVINO `2026.1`, PyTorch `2.12.0+xpu`, llama.cpp `b9404` (SYCL) |
| **Compiler** | oneAPI `2026.0` (IntelLLVM, MKL, DNNL, TBB) |
| **GPU driver** | `libze-intel-gpu1 26.14.37833.4` |

| Model | Task | Format | Size |
|-------|------|--------|------|
| **Qwen3-Embedding-4B** | Text embedding (2560-dim) | OpenVINO INT8 | ~3 GB |
| **Qwen3-Reranker-4B** | Text reranking | OpenVINO INT8 | ~3 GB |
| **Qwen3.5-4B** | Text generation | GGUF (IQ4_XS) | ~2.5 GB |

---

## The Problem (Observed with these models)

When running the above models on the described test platform, we observed the following issues:

- **Kernel Panic** – Complete system freeze requiring hard reboot (occurred during INT4 batch embedding)
- **Segfault (exit code 139)** – Process crashes from GPU memory access violation
- **NaN/Inf Output** – Model outputs garbage numerical values before crashes
- **Throttling** – Inference throughput drops 2–5× as the GPU downclocks under sustained load

**Possible explanation (observed on this platform):** Under our specific test conditions, the GPU driver became unstable when running at high utilization (>90%) for extended periods. This may be related to thermal/power management characteristics of this particular GPU+driver combination — but the exact root cause has not been independently verified on other hardware.

---

## Observations on Data Source

The GPU metrics in the comparison table below were collected using `intel_gpu_top` with 1-second sampling intervals, captured during active embedding inference. The "Before" data was captured before any crash-mitigation measures were applied; the "After" data was captured with the protection strategies described below. All metrics are from the same GPU under the same ambient conditions.

---

## Key Findings

> ⚠️ **Caveat:** The following findings are based on this specific test setup only. They may not generalize.

### 1. NaN Output Appeared as a Crash Warning (in this experiment)

In our tests, NaN/Inf values in inference output did **not** appear to be a quantization precision issue — instead, they consistently preceded GPU driver crashes. After implementing active cooling, NaN disappeared completely even with the same INT8 model. This suggests the root cause in our case was GPU overheating / power constraints, not numerical precision. Other users may have different experiences.

### 2. OpenVINO Used the Compute Engine on This Platform

On our test system, OpenVINO ran inference on the **Compute (CCS)** engine. When monitoring GPU utilization, `CCS` busy % was the relevant metric. This may vary by GPU architecture, driver version, or OpenVINO plugin implementation.

### 3. Crashes Were Caused by Sustained High Load, Not Quantization Format

In our experiments, both INT4 and INT8 configurations crashed under sustained high GPU utilization (>90%). The key variable was GPU load level and duration, not the quantization format. After introducing batch-size limits and cooldown intervals, the INT8 configuration ran stably. It remains unclear whether INT4 with the same protection measures would also be stable — quantization format may still play a secondary role.

### 4. Cross-Project Validation: Same Pattern Across Different Frameworks

This pattern was not limited to OpenVINO inference. We observed the same behavior in **two additional projects** on the same platform:

| Project | Framework | Workload | Issue | Mitigation | Result |
|---------|-----------|----------|-------|------------|--------|
| **Qwen3 embedding** | OpenVINO | LLM inference | Kernel panic / segfault | INT8 + batch=10 + cooling | ✅ Stable |
| **Gated CNN poetry** | PyTorch XPU | Text model training (61M params) | Terminal freeze, training stalls | Per-8-batch 3s cooling + per-update 5s cooling + latency monitoring | ✅ Stable (loss 9.1→0.85) |
| **AlphaZero gomoku** | PyTorch XPU | Game AI training (CNN) | Segfault at batch ~293 | Per-50-batch 30s cooling + retry ×3 | ✅ Stable |
| **ComfyUI T2I** | PyTorch XPU (ComfyUI) | Diffusion image generation | Black images at high res/steps | KSampler per-step 3s cooling + throttling detection | ✅ Stable (1440×1024, 16 steps) |

In all **four** cases, the fix was the same: **reduce sustained GPU load with frequent cooldown breaks.** The symptoms (terminal freezing, segfaults, NaN output) looked different but shared a common root.

### 6. Fourth Validation: ComfyUI Image Generation (2026-05-31)

The same protection pattern was successfully applied to ComfyUI (image generation):

| Project | Framework | Workload | Issue | Mitigation | Result |
|---------|-----------|----------|-------|------------|--------|
| **ComfyUI text-to-image** | PyTorch XPU (ComfyUI) | Diffusion image generation | Black images at high res / many steps | Per-step 3s cooling + throttling detection + `empty_cache` | ✅ Stable (1440×1024, 16 steps) |

**Before:** 1024×1024 at 8 steps was the maximum; higher resolutions or more steps produced completely black images (NaN output).
**After:** 1440×1024 at 16 steps generates correctly.

The fix was applied at the KSampler callback level (`comfy/samplers.py`), adding 3s cooldown with throttling detection after each diffusion step. Also added XPU environment variables to the launch script (`run_intel_gpu.sh`).

**Key implementation detail:** Unlike the other projects which use batch-level cooling, ComfyUI's cooling is at the **individual diffusion step level** — every step of the KSampler gets a brief rest. This is because each step is a full model evaluation (unet forward), comparable to a training batch in intensity.

### 5. Possible Root Cause: Software Protection Layer, Not Hardware

Based on our observations across these projects, the instability may be attributed to the **Intel XPU software stack lacking the protective mechanisms found in CUDA**, rather than a fundamental hardware defect:

- **CUDA** has built-in watchdog timers, graceful error recovery, driver timeout detection, and automatic context cleanup
- **Intel XPU** appears to lack several of these safeguards — when the GPU is under sustained load, the driver can hang, crash, or take down the entire system without warning

On NVIDIA platforms, equivalent workloads run stably without manual cooling because the driver handles thermal/power events gracefully. On this Intel platform, the same workloads require explicit cooldown management to stay within the driver's safe operating envelope. This does **not** mean the hardware is defective — it may simply reflect differences in software maturity.

> ⚠️ This is a hypothesis based on our specific test environment. We have not independently verified the absence of these mechanisms in the Intel XPU driver stack, nor do we claim that CUDA is universally superior for all workloads.

---

## Before / After Comparison

### GPU Workload Metrics

| Metric | Before (INT4, batch=20, no cooling) | After (INT8, batch=10, with cooling) |
|--------|-------------------------------------|--------------------------------------|
| GPU Compute (CCS) usage | > **90%** sustained | Peak **~19%**, avg **2.3%** |
| GPU freq (actual) | Sustained ~1800 MHz | Avg **456 MHz**, peak **2151 MHz** |
| GPU power | Sustained high → overheating | Avg **2.9 W**, peak **20.8 W** |
| RC6 idle ratio | ~**0%** (never rests) | **~31%** (frequent cooling breaks) |
| NaN/Inf output | ❌ Frequent → Kernel Panic | ✅ Zero NaN, zero crashes |
| Time per 1000 texts | ~**60 s** (fast but dangerous) | ~**350 s** (slow but safe) |

**Measurement method:** GPU metrics were captured via `intel_gpu_top` with 1-second sampling, over a 70-second continuous window during active embedding inference. The "Before" and "After" measurements were taken under identical ambient temperature conditions.

### Stability Comparison

| Config | Speed (1000 texts) | Stability | Verdict |
|--------|-------------------|-----------|---------|
| GPU INT4, batch=20 | ~60 s | ❌ Kernel panic | Do not use |
| **GPU INT8, batch=10 + cooling** | **~350 s** | **✅ Stable** | **Recommended** |
| CPU INT8, batch=10 | ~10 min | ✅ Rock solid | Fallback |
| NPU INT8 | TBD | Needs static shapes | Future |

---

## Cross-Project Protection Pattern

Across all three projects on this platform, a consistent protection strategy emerged:

```
┌─────────────────────────────────────────────────┐
│          Intel GPU Protection Pattern            │
├─────────────────────────────────────────────────┤
│  1. Active cooldown: short breaks at regular    │
│     intervals (every 8-50 batches)              │
│  2. Latency monitoring: track batch time        │
│     baseline, detect throttling (>2x baseline)  │
│  3. Escalated cooldown: extra rest when         │
│     throttling detected                         │
│  4. Crash retry: auto-retry with GPU reset      │
│  5. Checkpointing: frequent saves for resume    │
└─────────────────────────────────────────────────┘
```

This pattern is a workaround for what may be a **software protection gap** — the Intel GPU driver stack may not yet have the same level of fault tolerance as more mature platforms. With these measures, all three projects ran stably.

---

## Summary (for this specific setup)

On the tested platform and with the tested models, the following strategies helped achieve stable operation:

- **Small batches** appeared to prevent power spikes
- **Frequent breaks** between batches gave the GPU time to cool
- **Thermal detection via latency monitoring** caught throttling early and triggered extra cooldown
- **INT8 quantization** combined with load limits and cooldown intervals achieved stable operation (INT4 alone was not tested with these protections)

Without these measures, sustained GPU load >90% eventually triggered a driver crash regardless of quantization format. The same pattern appeared across three different projects (OpenVINO inference, PyTorch XPU training, AlphaZero training), suggesting a common root cause in the software protection layer rather than a hardware defect.

These findings may or may not apply to other Intel Arc configurations.

---

> *This document reflects observations from a single test environment. If you have similar or contradictory experiences on different hardware, contributions are welcome.*

---

## 观测方法

本文数据来自以下设置和测量方法：

- **工作负载：** 使用 Qwen3-Embedding-4B (OpenVINO) 批量编码中国古典诗歌文本（约 7000 首，每首 100–500 字）
- **监控工具：** `intel_gpu_top`（`intel-gpu-tools` 套件的一部分），每秒采样一次
- **监控时长：** Embedding 推理过程中连续采集 70 秒
- **崩溃检测：** 进程退出码监控（exit 139 = 段错误）、内核日志 (`dmesg`) 检测 Kernel Panic、输出张量检查 NaN/Inf
- **对比方式：** 同一模型、同一数据、同一 GPU——仅量化格式 (INT4 vs INT8)、批次大小 (20 vs 10) 和冷却策略不同

---

# Intel Arc GPU AI 负载稳定性指南

> ⚠️ **免责声明：** 以下观察基于特定的硬件/软件组合和特定模型（详见下方）。**不代表** Intel Arc GPU 的普遍行为，也不保证适用于其它配置。请作为个案参考，而非通用结论。

**在 Intel Arc 独立显卡上运行 Qwen3 系列模型时观察到的 GPU 驱动崩溃（Kernel Panic、段错误）问题记录与缓解方法。**

---

## 测试平台与模型

| 类别 | 详情 |
|------|------|
| **GPU** | Intel Arc Pro 130T/140T (Arrow Lake-P, PCI ID `8086:7d51`) |
| **NPU** | Intel AI Boost (`/dev/dri/renderD128`) |
| **内存** | 64 GB |
| **系统** | Ubuntu 26.04 LTS |
| **内核** | `7.0.0-15-generic` |
| **框架** | OpenVINO `2026.1`, PyTorch `2.12.0+xpu`, llama.cpp `b9404` (SYCL) |
| **编译器** | oneAPI `2026.0` (IntelLLVM, MKL, DNNL, TBB) |
| **GPU 驱动** | `libze-intel-gpu1 26.14.37833.4` |

| 模型 | 任务 | 格式 | 大小 |
|------|------|------|------|
| **Qwen3-Embedding-4B** | 文本向量化 (2560维) | OpenVINO INT8 | ~3 GB |
| **Qwen3-Reranker-4B** | 文本排序 | OpenVINO INT8 | ~3 GB |
| **Qwen3.5-4B** | 文本生成 | GGUF (IQ4_XS) | ~2.5 GB |

---

## 问题描述（在本次测试中观察到）

在运行上述模型时，我们观察到了以下问题：

- **Kernel Panic** — 系统完全死机，需要硬重启
- **段错误 (exit code 139)** — 进程崩溃，显存访问异常
- **NaN/Inf 输出** — GPU 在崩溃前输出异常值
- **降频** — 推理速度骤降 2-5 倍

**可能原因（仅限本次测试）：** 在本测试环境下，GPU 驱动在高占用率（>90%）下持续运行后变得不稳定。具体原因可能与特定 GPU+驱动版本组合的热管理特性有关——尚未在其它硬件上独立验证。

---

## 关键发现

> ⚠️ **注意：** 以下发现仅基于本次测试环境，未必具有普遍性。

### 1. NaN 输出是该实验中的崩溃前兆

在本测试中，NaN/Inf 输出**并非量化精度问题**，而是 GPU 驱动即将崩溃的前兆。加入主动冷却后，同一 INT8 模型的 NaN 完全消失。说明我们遇到的根因是 GPU 过热/供电受限，而非数值精度。其它配置下可能不同。

### 2. 本次测试中 OpenVINO 使用 Compute 引擎

在我们的系统上，OpenVINO 的推理走 **Compute (CCS)** 引擎。监控时应关注 `CCS` 占用率。不同 GPU 架构、驱动版本或 OpenVINO 插件实现可能不同。

### 3. 崩溃由持续高负载引起，而非量化格式本身

在我们的实验中，INT4 和 INT8 配置在持续高 GPU 占用率（>90%）下均出现崩溃。关键变量是 GPU 负载水平和持续时间，而非量化格式。加入 batch 限制和冷却间隔后，INT8 配置实现了稳定运行。目前尚不清楚 INT4 在同等保护措施下是否也能稳定——量化格式可能起次要作用。

### 4. 跨项目验证：不同框架下的相同规律

这种现象不仅出现在 OpenVINO 推理中。我们在同一平台上的**另外两个项目**中观察到了相同的行为：

| 项目 | 框架 | 工作负载 | 问题 | 缓解措施 | 结果 |
|------|------|---------|------|---------|------|
| **Qwen3 向量化** | OpenVINO | 大模型推理 | Kernel panic / 段错误 | INT8 + batch=10 + 冷却 | ✅ 稳定 |
| **Gated CNN 诗词** | PyTorch XPU | 文本模型训练 (61M) | 终端卡死、训练中断 | 每8批冷3s + 每更新冷5s + 耗时检测 | ✅ 稳定 (loss 9.1→0.85) |
| **AlphaZero 五子棋** | PyTorch XPU | 游戏 AI 训练 (CNN) | 约第293批段错误 | 每50批冷30s + 重试×3 | ✅ 稳定 |
| **ComfyUI 文生图** | PyTorch XPU (ComfyUI) | 扩散模型图像生成 | 高分辨率/多步数出黑图 | KSampler 每步冷3s + 降频检测 | ✅ 稳定 (1440×1024, 16步) |

四个案例的解决方案相同：**通过频繁冷却来降低 GPU 持续负载。** 症状（终端冻结、段错误、NaN输出）看起来不同，但根源可能一致。

### 5. 可能原因：软件保护层不到位，而非硬件问题

综合多个项目的观察，这些问题可能源于 **Intel XPU 软件栈缺少 CUDA 中已有的保护机制**，而非硬件缺陷：

- **CUDA** 内置了 watchdog 定时器、优雅错误恢复、驱动超时检测和自动上下文清理
- **Intel XPU** 可能缺少部分上述保护——当 GPU 持续高负载时，驱动可能直接挂起、崩溃或不告警地拖垮整个系统

在 NVIDIA 平台上，同等负载无需手动冷却即可稳定运行，因为驱动本身处理了热管理/功耗事件。而在此平台上，同等负载需要明确的冷却策略才能保持在驱动的安全运行范围内。这**不意味着硬件有缺陷**——可能只是反映了软件成熟度的差异。

> ⚠️ 这仅是基于我们特定测试环境的推测。我们未独立验证 Intel XPU 驱动栈是否确实缺少这些机制，也不认为 CUDA 在所有场景下都更优越。

---

## 优化前后对比

### GPU 负载指标

| 指标 | 优化前 (INT4, batch=20, 无冷却) | 优化后 (INT8, batch=10, 有冷却) |
|------|--------------------------------|-------------------------------|
| GPU Compute (CCS) 占用 | > **90%** 持续满载 | 峰值 **~19%**，均值 **2.3%** |
| GPU 实际频率 | 持续 ~1800 MHz | 均值 **456 MHz**，峰值 **2151 MHz** |
| GPU 功耗 | 持续高负载 → 过热 | 均值 **2.9 W**，峰值 **20.8 W** |
| RC6 空闲比例 | ~**0%**（从不休息） | **~31%**（频繁冷却） |
| NaN/Inf 输出 | ❌ 频繁出现 → Kernel Panic | ✅ 零 NaN、零崩溃 |
| 每千条耗时 | ~**60 s**（快但不稳） | ~**350 s**（慢但安全） |

**测量方法：** GPU 指标通过 `intel_gpu_top` 以 1 秒间隔采集，在 Embedding 推理过程中连续采样 70 秒。优化前后的测量在相同室温条件下进行。

### 稳定性对比

| 配置 | 速度 (千条) | 稳定性 | 结论 |
|------|-----------|--------|------|
| GPU INT4, batch=20 | ~60 s | ❌ Kernel panic | 不推荐 |
| **GPU INT8, batch=10 + 冷却** | **~350 s** | **✅ 稳定** | **推荐** |
| CPU INT8, batch=10 | ~10 min | ✅ 绝对稳定 | 备用 |
| NPU INT8 | 待测 | 需静态 shape | 后续 |

---

> 注：以下策略在三个不同项目（OpenVINO 推理、PyTorch XPU 训练、AlphaZero 训练）中得到验证，可能具有一定的通用参考价值。

## 跨项目通用保护模式

基于三个项目的经验，总结出一套在本次平台上有效的 GPU 保护策略：

```
┌─────────────────────────────────────────────────┐
│           Intel GPU 保护策略通用模板               │
├─────────────────────────────────────────────────┤
│  1. 主动冷却：每隔若干 batch 加入短暂休息         │
│     (8-50 batch 间隔，视 GPU 负载定)              │
│  2. 耗时监控：追踪 batch 耗时基线                  │
│     检测降频（>2x 基线时触发额外冷却）             │
│  3. 分级冷却：轻度降频冷 10s，严重降频冷 20s       │
│  4. 崩溃重试：自动捕获异常，清理显存后重试 ×3       │
│  5. 断点保存：每次参数更新保存 checkpoint          │
└─────────────────────────────────────────────────┘
```

这套策略本质上是在弥补可能存在的 **软件保护层缺口**——Intel GPU 驱动栈的容错机制可能不及更成熟的平台完善。加入这些措施后，三个项目均稳定运行。

## 总结（仅限于本次测试环境）

在本次测试的平台和模型下，以下策略帮助实现了稳定运行：

- **小 batch** 可能避免了瞬时功耗尖峰
- **频繁间歇** 让 GPU 有充分冷却时间
- **通过推理延迟监测温度**，在降频初期及时触发额外冷却
- **INT8 量化 + 负载限制 + 冷却间隔** 的组合实现了稳定运行（尚未在同等保护下测试 INT4）

如果不采取这些措施，无论采用何种量化格式，GPU 在 >90% 占用率下持续运行最终触发了驱动崩溃。这一规律在三个不同项目（OpenVINO 推理、PyTorch XPU 训练、AlphaZero 训练）中均得到重现，可能指向软件保护层的共性问题，而非硬件缺陷。

这些发现不保证适用于其它 Intel Arc 配置。

---

> *本文档仅反映单一测试环境下的观察结果。如果你在其它硬件上有类似或相反的体验，欢迎提供反馈。*
