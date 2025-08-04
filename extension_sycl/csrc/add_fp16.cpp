#include <stdlib.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#include "sycl/sycl.hpp"

namespace {

template<typename T>
void add_impl(const sycl::nd_item<2>& item, void* a, void* b, void* c,
        uint32_t input_size, int32_t actual_work_items, int32_t work_group_size_cols) {
    auto local_id = item.get_local_id(1);
    auto group_id = item.get_group(1);

    T* a_ptr = reinterpret_cast<T*>(a);
    T* b_ptr = reinterpret_cast<T*>(b);
    T* c_ptr = reinterpret_cast<T*>(c);

    // start address of the current work group
    size_t start = group_id * work_group_size_cols;
    for (size_t i = start + local_id; i < input_size; i += actual_work_items) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

} // namespace

namespace extension_sycl {

at::Tensor add_fp16(const at::Tensor &a, const at::Tensor &b) {
    TORCH_CHECK(a.dtype() == at::kHalf, "Input tensor 'a' must be of type float16");
    TORCH_CHECK(b.dtype() == at::kHalf, "Input tensor 'b' must be of type float16");
    // can also support broadcast
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");

    auto stream = at::xpu::getCurrentXPUStream();
    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kXPU);

    at::Tensor c = torch::empty(a.sizes(), options);

    void* a_ptr = a.data_ptr();
    void* b_ptr = b.data_ptr();
    void* c_ptr = c.data_ptr();

    sycl::queue& queue = stream.queue();

    auto input_size = a.numel();
    // number of elements processed in a single work group
    int work_group_size = std::min<int>(1024, input_size);
    // 2D work group size
    int work_group_size_rows = 1;
    int work_group_size_cols = work_group_size;
    // total processed work items???
    // how many subslices?
    // how many items can be processed in a subslice?
    int total_num_work_items = input_size;
    int work_group_col_size = work_group_size_cols;
    int grid_size_rows = 1;
    int grid_size_cols = (total_num_work_items + work_group_col_size - 1) / work_group_col_size;
    int actual_work_items = grid_size_cols * work_group_col_size;

    sycl::range<2> global_range(grid_size_rows * work_group_size_rows, grid_size_cols * work_group_size_cols);
    sycl::range<2> block_range(work_group_size_rows, work_group_size_cols);
    sycl::nd_range<2> nd_range(global_range, block_range);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class add_kernel>(nd_range, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]] {
            add_impl<sycl::half>(item, a_ptr, b_ptr, c_ptr, total_num_work_items, actual_work_items, work_group_col_size);
        });
    }).wait();

    return c;
}


TORCH_LIBRARY(extension_sycl, m) {
    m.def("add_fp16(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_sycl, XPU, m) {
    m.impl("add_fp16", &extension_sycl::add_fp16);
}

} // namespace extension_sycl
