# Copyright (c) 2017 Manzil Zaheer All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import platform

PACKAGE_NAME = 'covertree'

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Modern numpy doesn't need the old __NUMPY_SETUP__ workaround
        import numpy
        self.include_dirs.append(numpy.get_include())

# Platform-specific compiler flags
# Note: Using C++17 for shared_mutex support on macOS
extra_compile_args = ['-pthread', '-std=c++17', '-O3', '-Wno-deprecated-declarations']
extra_link_args = ['-pthread', '-std=c++17']

if sys.platform == 'darwin':  # macOS
    print(f"Configuring build for macOS ({platform.machine()})")

    # Use clang-specific flags
    extra_compile_args.extend(['-stdlib=libc++'])
    extra_link_args.extend(['-stdlib=libc++'])

    # Optimize for the actual hardware
    machine = platform.machine()
    if machine == 'arm64':
        # Apple Silicon (M1/M2/M3/M4)
        print("  Detected Apple Silicon - using -mcpu=apple-m1")
        extra_compile_args.append('-mcpu=apple-m1')
        extra_link_args.append('-mcpu=apple-m1')
    else:
        # Intel Mac
        print("  Detected Intel Mac - using -march=native")
        extra_compile_args.append('-march=native')
        extra_link_args.append('-march=native')
elif sys.platform == 'linux':
    # Linux (original behavior)
    print("Configuring build for Linux")
    extra_compile_args.append('-march=corei7-avx')
    extra_link_args.append('-march=corei7-avx')
else:
    # Windows or other
    print(f"Configuring build for {sys.platform}")
    # Use conservative flags
    pass

# Old C API module (for backward compatibility)
covertreec_module = Extension(
    'covertreec',
    sources=[
        'src/cover_tree/covertreecmodule.cxx',
        'src/cover_tree/cover_tree.cpp'
    ],
    include_dirs=['lib/'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++'
)

# New pybind11 module (modern bindings)
covertree2_module = Extension(
    'covertree2',
    sources=[
        'src/cover_tree/covertree_pybind.cpp',
        'src/cover_tree/cover_tree.cpp'
    ],
    include_dirs=[
        'lib/',
        'lib/pybind11/include'
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++'
)

# If pyproject.toml exists, setuptools will read config from there
# This setup.py is now primarily for the C++ extension build
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[covertreec_module, covertree2_module],
)


