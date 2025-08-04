# Introduction

Refer to https://github.com/pytorch/extension-cpp.git

An example of writing Sycl extension op for PyTorch.

# Build

```shell
# setup oneAPI env
source /opt/intel/oneapi/setvars.sh

# WA to avoid cutlass build issue
cd third-party/cutlass-sycl
patch -p1 < ../../patch_wa_cutlass.diff

pip install --no-build-isolation -e .
```

# Test

```shell
python tests/test_gemm.py
```
