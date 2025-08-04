import torch
import os

import extension_sycl

a = torch.randn(1240, dtype=torch.float16, device="xpu")
b = torch.randn(1240, dtype=torch.float16, device="xpu")

ref = torch.ops.aten.add(a, b)
print("ref: ", ref)

actual = torch.ops.extension_sycl.add_fp16(a, b)

print("actual: ", actual)
assert torch.allclose(ref.cpu(), actual.cpu())
