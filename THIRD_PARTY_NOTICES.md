# Third-party notices

`_patches/torch_mpp_qcache.py` expands the locally installed PyTorch 2.14
`MppAttention.h` / `PrefillAttention.h` and their Metal utility headers, checks
their combined source hash, and moves query loads outside the key loop.
The validated PyTorch commit is `08187d9e0fba026dc8217405802ab5381dc88d90`.
These attention kernels are adapted from MLX. PyTorch and MLX license notices
are reproduced in [_patches/krea2_ext/PYTORCH_LICENSE](_patches/krea2_ext/PYTORCH_LICENSE)
and [_patches/krea2_ext/MLX_LICENSE](_patches/krea2_ext/MLX_LICENSE).

The GroupNorm reduction/affine ordering in
`_patches/seedvr2_ext/seedvr2_kernels.mm` is adapted from PyTorch's
`aten/src/ATen/native/mps/kernels/GroupNorm.metal` and
`c10/metal/reduction_utils.h` at commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

PyTorch's copyright notices and license terms are reproduced in
[_patches/seedvr2_ext/PYTORCH_LICENSE](_patches/seedvr2_ext/PYTORCH_LICENSE).

The fixed SeedVR2 conditioning tensors in
`_patches/seedvr2_assets/conditioning.safetensors` come from
[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185)
(`pos_emb.pt` and `neg_emb.pt`). Only the container and tensor names changed;
the BF16 tensor bytes are preserved. Their Apache 2.0 license is reproduced
in [_patches/seedvr2_assets/LICENSE](_patches/seedvr2_assets/LICENSE).

The RMSNorm reduction ordering in `_patches/krea2_ext/krea2_kernels.mm`
follows PyTorch's `aten/src/ATen/native/mps/kernels/RMSNorm.metal`
at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`, which is adapted
from MLX's `mlx/backend/metal/kernels/rms_norm.metal`
(Copyright © 2024 Apple Inc.). The original license notices are included in
[_patches/krea2_ext/PYTORCH_LICENSE](_patches/krea2_ext/PYTORCH_LICENSE)
and [_patches/krea2_ext/MLX_LICENSE](_patches/krea2_ext/MLX_LICENSE).

The Qwen-Image 2.1 RMSNorm reduction in `_patches/qwen21_fast.metal`
follows the same PyTorch/MLX reduction convention as the Krea2 kernels above.
The PyTorch and MLX license notices are included in `_patches/krea2_ext`.

The tile grouping and attention loop in `_patches/ltx25_vae.py` are adapted
from Comfy Kitchen 0.2.35 `backends/eager/na.py`, Copyright (c) 2025 Comfy Org.
All rights reserved. Changes batch independent MPS tiles under a bounded
workspace, fuse neighborhood copies with bounded integer row maps, and build
identical masks without GPU readback. Apache 2.0 license
terms are reproduced in [_patches/seedvr2_assets/LICENSE](_patches/seedvr2_assets/LICENSE).
The LTX RMSNorm reduction in `_patches/ltx25_fast.metal` follows the same
PyTorch/MLX convention and license notices referenced above.
