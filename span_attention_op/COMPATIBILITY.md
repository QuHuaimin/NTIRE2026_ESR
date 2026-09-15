# Local build compatibility

The CUDA kernels originate from the official NTIRE 2026 Team 22 submission.
The local workstation uses PyTorch 2.4.1, CUDA 11.8 and GCC 13 by default, so
three build-only compatibility changes are applied:

- use GCC/G++ 11, which CUDA 11.8 supports;
- pass PyTorch include directories as a flat list;
- reinterpret `c10::Half` storage as CUDA `half` pointers explicitly, avoiding
  ambiguous constructors in newer PyTorch headers.

The attention equations, channel specializations, launch dimensions and
fast-math flags are unchanged. On an RTX 4060, an end-to-end FP32 comparison
with the official checkpoint and a seeded 64x64 input produced a mean absolute
difference of 0.00106 (maximum 0.03755) between the fused and pure PyTorch
paths. The fused operator is therefore kept for official inference timing only;
training uses the differentiable pure PyTorch path (`use_span_attn: false`).
