# Introduction

Refer to https://github.com/pytorch/extension-cpp.git

An example of writing Sycl extension op for PyTorch.

# Build and Install

```shell
# create virtual env
# install pytorch xpu package
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

# setup oneAPI env, require version 2025.1
source /opt/intel/oneapi/setvars.sh

pip install --no-build-isolation -e .

```

# Test

```python
# add below code to load extension
import extension_sycl
```

```shell
cd tests
python tests/test_gemm.py
```
