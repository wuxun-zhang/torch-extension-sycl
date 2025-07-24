
# Adapted from https://github.com/pytorch/extension-cpp/blob/master/setup.py
# this is a setup.py script to build the Sycl/C++ extension for PyTorch.
import os
import glob
import torch

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    SyclExtension,
    BuildExtension,
    _COMMON_SYCL_FLAGS
)

library_name = "extension_sycl"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    extension = CppExtension
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    extra_cutlass_compile_flags="-DCUTLASS_ENABLE_SYCL;-DSYCL_INTEL_TARGET;-DNDEBUG;".split(";")

    extra_compile_args = {
        "cxx": ["-O3" if not debug_mode else "-O0 -g", "-std=c++17"] + extra_cutlass_compile_flags,
    }

    _COMMON_SYCL_FLAGS.extend(extra_cutlass_compile_flags)
    # extra_cutlass_flags = "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier;-fno-sycl-instrument-device-code;-ftemplate-backtrace-limit=0;-v".split(";")
    # _COMMON_SYCL_FLAGS.extend(extra_cutlass_flags)

    extra_link_args = []
    if debug_mode:
        extra_link_args.extend(["-g", "-O0"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name)
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_sycl_dir = os.path.join(extensions_dir, "csrc")
    cpp_sources = list(glob.glob(os.path.join(extensions_sycl_dir, "*.cpp")))
    # find subdirectories for additional sources
    if os.listdir(extensions_sycl_dir):
        cpp_sources.extend(
            glob.glob(os.path.join(extensions_sycl_dir, "*", "*.cpp"))
        )
    # print(f"Found C++ sources: {cpp_sources}")
    sycl_sources = []
    # WA: generate .sycl files from .cpp files
    # PyTorch cpp extension tool requires .sycl suffix to trigger SYCL
    # compilation
    for source in cpp_sources:
        sycl_source = source.replace(".cpp", ".sycl")
        # always generate new sycl file based on cpp source file
        with open(source, "r") as f:
            content = f.read()
        with open(sycl_source, "w") as f:
            f.write(content)
        sycl_sources.append(sycl_source)

    if sycl_sources:
        sources.extend(sycl_sources)
        # mix compilation of C++ and SYCL sources
        extension = SyclExtension

    include_dirs = []
    # include_dirs.append(os.path.abspath(os.path.join(this_dir, "include")))
    # add cutlass headers
    include_dirs.append(os.path.abspath(os.path.join(this_dir, "third-party/cutlass-sycl/include")))
    include_dirs.append(os.path.abspath(os.path.join(this_dir, "third-party/cutlass-sycl/examples/common")))
    include_dirs.append(os.path.abspath(os.path.join(this_dir, "third-party/cutlass-sycl/tools/util/include")))

    print(include_dirs)
    return [
        extension(
            name=f"{library_name}._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            py_limited_api=py_limited_api,
        )
    ]

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example sycl extension for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wuxun-zhang/torch-extension-sycl.git",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

