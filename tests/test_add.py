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

a = torch.randn(1240, dtype=torch.float16, device="xpu")
b = torch.randn(1240, dtype=torch.float16, device="xpu")

ref = torch.ops.aten.add(a, b)
print("ref: ", ref)

actual = torch.ops.extension_sycl.add_fp16(a, b)

print("actual: ", actual)
assert torch.allclose(ref.cpu(), actual.cpu())
