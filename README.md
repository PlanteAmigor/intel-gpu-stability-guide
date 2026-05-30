# Intel GPU AI Training & Inference Issue Logs

[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2026.04%20LTS-E95420)](https://ubuntu.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+xpu-EE4C2C)](https://pytorch.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.1-00A3E0)](https://docs.openvino.ai/)

This directory documents issues encountered during AI training and inference on a specific Intel platform (Arrow Lake + Arc GPU + NPU), along with mitigation strategies explored.

> ⚠️ **Disclaimer:** All observations here are based on specific hardware/software/model combinations and **may not apply to other environments**. See detailed disclaimers in each document.

🌐 **Language:** [English](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/README.md) · [中文](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/README.zh-CN.md)

---

## Hardware Environment

| Component | Model |
|-----------|-------|
| **CPU** | Intel Core Ultra 9 285H (Arrow Lake, 6P+8E, 14 cores) |
| **iGPU** | Intel Arc Graphics (8 Xe-core, 128 GB shared memory) |
| **dGPU** | Intel Arc Pro 130T/140T (Arrow Lake-P, PCI ID `8086:7d51`) |
| **NPU** | Intel AI Boost (`/dev/dri/renderD128`) |
| **RAM** | 128 GB DDR5 (shared with iGPU) |

## Software Environment

| Software | Version |
|----------|---------|
| **OS** | Ubuntu 26.04 LTS (Linux Kernel 7.0) |
| **PyTorch** | 2.12.0+xpu |
| **OpenVINO** | 2026.1 |
| **Python** | 3.14 |
| **oneAPI** | 2026.0 (IntelLLVM, MKL, DNNL, TBB) |
| **GPU driver** | `libze-intel-gpu1 26.14.37833.4` |

---

## Documents

### 1. [dGPU Stability Guide](intel_gpu_aitrain001.md)

**Use case:** OpenVINO + Qwen3-series model inference on Intel Arc discrete GPU.

**Key findings (on this platform):**
- Sustained GPU utilization >90% → Kernel Panic / segfault
- NaN output → precursor to driver crash (not a quantization precision issue)
- Mitigation: small batch + frequent cooldown intervals + INT8 quantization

| File | Description |
|------|-------------|
| [intel_gpu_aitrain001.md](intel_gpu_aitrain001.md) | Full analysis (bilingual) |

---

### 2. [iGPU Transformer Backward Crash Report](xpu_backward_issue.md)

**Use case:** PyTorch XPU backend Transformer training on Arc iGPU.

**Key findings (on this platform):**
- `nn.TransformerEncoderLayer` / `F.scaled_dot_product_attention` backward pass may crash
- Error types: `RuntimeError` (negative dimension / integer overflow) or `IndexError` (index out of range)
- Even tiny 22M-parameter models (hidden=512, heads=8, batch=4) may crash
- AMP BF16 may escalate to full system freeze (driver crash)
- Attention-free architectures (Gated CNN, Conv2d) run stably

| File | Description |
|------|-------------|
| [xpu_backward_issue.md](xpu_backward_issue.md) | Full analysis (bilingual) |

---

## Summary

| Scenario | GPU | Framework | Status | Recommended Alternative |
|----------|-----|-----------|--------|------------------------|
| **Transformer training** | iGPU | PyTorch XPU | ❌ Backward crash | Gated CNN / CPU training |
| **CNN training** | iGPU | PyTorch XPU | ✅ Stable | — |
| **LLM inference (Qwen3)** | dGPU | OpenVINO | ⚠️ Needs cooling | INT8 + small batch + cooldown |
| **LLM inference** | NPU | OpenVINO | ✅ Tested | Requires static shapes |
| **LLM inference** | CPU | OpenVINO | ✅ Stable | Slowest performance |

---

## External Links

- [Intel GPU Stability Guide (GitHub)](https://github.com/PlanteAmigor/intel-gpu-stability-guide) — External mirror of this repo
- [PyTorch XPU Documentation](https://docs.pytorch.org/docs/2.12/xpu.html)
- [OpenVINO Documentation](https://docs.openvino.ai/)

---

*These documents record observations from a specific test environment for reference only. Different hardware, driver, or framework versions may yield different results.*
