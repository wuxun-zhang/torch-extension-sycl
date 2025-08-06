import torch

import extension_sycl

M = 5120
N = 4096
K = 4096

A = torch.randn(5120, 4096, dtype=torch.bfloat16, device="xpu")
B = torch.randn(4096, 4096, dtype=torch.bfloat16, device="xpu")

ref_xpu = torch.matmul(A, B)

C = torch.zeros(5120, 4096, dtype=torch.float, device="xpu")
actual_xpu = torch.ops.extension_sycl.cutlass_gemm(A, B, C).to(torch.bfloat16)

torch.testing.assert_close(ref_xpu.cpu(), actual_xpu.cpu())
print("Correctness check passed.")

start = torch.xpu.Event(enable_timing=True)
end = torch.xpu.Event(enable_timing=True)

# warmup
for _ in range(3):
    torch.ops.extension_sycl.cutlass_gemm(A, B, C)

start.record()
for i in range(100):
    torch.ops.extension_sycl.cutlass_gemm(A, B, C)
end.record()
torch.xpu.synchronize()

elapsed_time = start.elapsed_time(end)
print(f"Benchmarking {M}x{N}x{K} GEMM on XPU")
print(f" Average time for 100 iterations: {elapsed_time / 100:.2f} ms")
total_flops = 2 * M * N * K
tflops = total_flops / (elapsed_time / 100 / 1000) * 1e-12
print(f" Performance: {tflops:.2f} TFLOPS")
