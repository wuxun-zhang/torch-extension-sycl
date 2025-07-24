import torch
import os

# option1: using package built by setup.py
import extension_sycl
###

# option2: using package built by CMake
# path = os.path.dirname(os.path.abspath(__file__))
# build_path = os.path.join(path, "..", "build")
# torch.ops.load_library(os.path.join(build_path, "extension_sycl.so"))
##

A = torch.randn(32, 128, dtype=torch.bfloat16, device="xpu")
B = torch.randn(128, 64, dtype=torch.bfloat16, device="xpu")

ref_xpu = torch.matmul(A, B).to(torch.float)

C = torch.zeros(32, 64, dtype=torch.float, device="xpu")
actual_xpu = torch.ops.extension_sycl.cutlass_gemm(A, B, C)

torch.testing.assert_close(ref_xpu.cpu(), actual_xpu.cpu())

