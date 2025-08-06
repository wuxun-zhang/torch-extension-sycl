#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
// #include <Python.h>
#include <torch/library.h>

namespace extension_sycl {
at::Tensor add_fp16(const at::Tensor &a, const at::Tensor &b);

// gemm
at::Tensor cutlass_gemm(const at::Tensor& A,
                           const at::Tensor& B,
                           const std::optional<at::Tensor> &out);


} // namespace extension_sycl
