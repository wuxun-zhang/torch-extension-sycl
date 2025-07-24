# Introduction

Refer to https://github.com/pytorch/extension-cpp.git

An example of writing Sycl extension op for PyTorch.

# Build

```shell
# setup oneAPI env
source /opt/intel/oneapi/setvars.sh

pip install --no-build-isolation -e .
```

# Test

```shell
python tests/test_gemm.py
```
