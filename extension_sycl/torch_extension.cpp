#include <Python.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "custom_ops.hpp"

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(extension_sycl, m) {
    m.def("cutlass_gemm(Tensor A, Tensor B, Tensor? C) -> Tensor");
    m.def("add_fp16(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_sycl, XPU, m) {
    m.impl("cutlass_gemm", &extension_sycl::cutlass_gemm);
    m.impl("add_fp16", &extension_sycl::add_fp16);
}
