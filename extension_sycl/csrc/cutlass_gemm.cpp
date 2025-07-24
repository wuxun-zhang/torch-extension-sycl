#include <c10/xpu/XPUStream.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <ATen/ATen.h>

#include "cutlass_gemm.hpp"

namespace extension_sycl {

template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get the input shapes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0]; 
  const int N = B.sizes()[1];

  // We cast the pointers to the type we need. We work with pointers instead of accessors because CUTLASS requires pointers.
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  OutputType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}

// Intermediate function to get the output precision to use for the wrapper template. 
template<typename DataType>
void cutlass_gemm_find_output_type(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  if(C.dtype() == torch::kFloat)
    cutlass_gemm_unpack<DataType, float>(A, B, C);
  // else if (C.dtype() == torch::kBFloat16)
  //   cutlass_gemm_unpack<DataType, bfloat16_t>(A, B, C);
  else
    throw std::invalid_argument("Unsupported precision type");
}

at::Tensor cutlass_gemm(const at::Tensor& A,  // A matrix (m x k)
                           const at::Tensor& B,  // B matrix (k x n)
                           const std::optional<at::Tensor>& out) {   // optional out matrix (m x n)

  // Handling the optional C matrix.
  at::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input.
    auto c_options = at::TensorOptions().device(at::kXPU).dtype(A.dtype());
    C = at::empty({M, N}, c_options);
  }

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_xpu() && B.device().is_xpu() && C.device().is_xpu()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use .to(device=torch.device('xpu'))");

  // Ensuring that the matrices are contiguous. 
  at::Tensor _A = A.contiguous();
  at::Tensor _B = B.contiguous();
  at::Tensor _C = C.contiguous();

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(_A.dtype() == at::kBFloat16)
    cutlass_gemm_find_output_type<cute::bfloat16_t>(_A, _B, _C);
  else
    throw std::invalid_argument("Unsupported input data type");

  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}

TORCH_LIBRARY(extension_sycl, m) {
    m.def("cutlass_gemm(Tensor A, Tensor B, Tensor? C) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_sycl, XPU, m) {
    m.impl("cutlass_gemm", &custom_xpu_extension::cutlass_gemm);
}

} // namespace extension_sycl

