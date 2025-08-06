import torch
import os

# option1: using package built by setup.py
import extension_sycl
###

# option2: using package built by CMake
# path = os.path.dirname(os.path.abspath(__file__))
# build_path = os.path.join(path, "..", "_build")
# torch.ops.load_library(os.path.join(build_path, "extension_sycl.cpython-312-x86_64-linux-gnu.so"))
##

A = torch.randn(32, 128, dtype=torch.bfloat16, device="xpu")
B = torch.randn(128, 64, dtype=torch.bfloat16, device="xpu")

ref_xpu = torch.matmul(A, B)

C = torch.zeros(32, 64, dtype=torch.float, device="xpu")
actual_xpu = torch.ops.extension_sycl.cutlass_gemm(A, B, C).to(torch.bfloat16)

torch.testing.assert_close(ref_xpu.cpu(), actual_xpu.cpu())
print("Test passed.")
