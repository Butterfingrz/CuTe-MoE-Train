"""Build for the group_gemm CUDA extension (bf16 grouped GEMM kernels)."""

import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

PACKAGE_DIR = Path(__file__).parent

# Wrap host compiler + nvcc with sccache when available so rebuilds (especially
# cutlass instantiations) hit the cache. -ccbin only accepts a single executable,
# so wrap "sccache <compiler>" in tiny shell scripts and point CXX/nvcc at those.
_sccache = shutil.which("sccache")
if _sccache and not os.environ.get("DISABLE_SCCACHE"):
    _wrapper_dir = PACKAGE_DIR / ".sccache_wrappers"
    _wrapper_dir.mkdir(exist_ok=True)

    def _make_wrapper(name: str, target: str) -> Path:
        path = _wrapper_dir / name
        path.write_text(f'#!/bin/sh\nexec {_sccache} {target} "$@"\n')
        path.chmod(0o755)
        return path

    # Note: leave CC alone — torch passes it as nvcc's -ccbin, and nvcc's
    # host-compiler property preprocessing rejects shell-script wrappers.
    _real_cxx = shutil.which(os.environ.get("CXX", "g++").split()[-1]) or "g++"
    _real_nvcc = shutil.which("nvcc") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"
    )

    os.environ["CXX"] = str(_make_wrapper("cxx", _real_cxx))
    _nvcc_wrapper = _make_wrapper("nvcc", _real_nvcc)

    # torch cpp_extension resolves nvcc via absolute path from CUDA_HOME.
    import torch.utils.cpp_extension as _cppext

    _orig_join = _cppext._join_cuda_home

    def _patched_join(*parts):
        if parts == ("bin", "nvcc"):
            return str(_nvcc_wrapper)
        return _orig_join(*parts)

    _cppext._join_cuda_home = _patched_join

# CUTLASS headers are staged into the package directory by the workspace task
# `prepare-grouped-mm-sources` before `pixi build` is invoked. We deliberately
# do not reach outside the package (the pixi-build sandbox does not see the
# repo's third-party/ tree).
cutlass_include = PACKAGE_DIR / "external" / "cutlass" / "include"
cutlass_tools_util_include = PACKAGE_DIR / "external" / "cutlass" / "tools" / "util" / "include"
if not (cutlass_include / "cutlass" / "cutlass.h").is_file():
    sys.exit(
        f"CUTLASS headers not found under {cutlass_include}. "
        "Run `pixi run prepare-grouped-mm-sources` from the workspace root "
        "to stage third-party/cutlass before building."
    )

nvcc_arch_flags = [
    "-gencode",
    "arch=compute_90a,code=sm_90a",
    "-gencode",
    "arch=compute_90a,code=compute_90a",
    "-D__CUDA_ARCH_FEAT_SM90_ALL",
]

setup(
    name="group_gemm",
    version="0.2.0",
    packages=["group_gemm"],
    package_dir={"group_gemm": "."},
    ext_modules=[
        CUDAExtension(
            name="group_gemm._ext",
            sources=["csrc/binding.cu"],
            include_dirs=[str(cutlass_include), str(cutlass_tools_util_include)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--std=c++17",
                    "--expt-relaxed-constexpr",
                    "-ftemplate-backtrace-limit=0",
                    "-Xcompiler=-fno-strict-aliasing",
                    "-Xcompiler=-Wconversion",
                ]
                + nvcc_arch_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
