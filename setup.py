import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
import glob

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

library_name = "extension_sycl"

def is_ninja_available() -> bool:
    return which("ninja") is not None

class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

# Adpated from implementations in https://github.com/vllm-project/vllm/blob/main/setup.py

class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        try:
            # os.sched_getaffinity() isn't universally available, so fall
            #  back to os.cpu_count() if we get an error here.
            num_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            num_jobs = os.cpu_count()

        return num_jobs

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(default_cfg),
        ]

        verbose = False
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        # if is_sccache_available():
        #     cmake_args += [
        #         '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
        #         '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
        #         '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
        #         '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
        #     ]
        # elif is_ccache_available():
        #     cmake_args += [
        #         '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
        #         '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
        #         '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
        #         '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
        #     ]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        # fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        # fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        # cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs = self.compute_num_jobs()

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix(f"{library_name}.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            prefix = self.build_lib

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


ext_modules = [CMakeExtension(f"{library_name}._C")]

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext": cmake_build_ext,
    }

setup(
    name=library_name,
    version="0.0.1",
    ext_modules=ext_modules,
    install_requires=["torch"],
    extras_require={},
    description="Sycl extension for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wuxun-zhang/torch-extension-sycl",
    cmdclass=cmdclass,
)
