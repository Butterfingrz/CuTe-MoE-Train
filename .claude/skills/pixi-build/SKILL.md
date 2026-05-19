---
name: pixi-build
description: "Package a PyTorch C++/CUDA extension as a standalone conda package using pixi-build-python. Use when adding a new CUDA op subdir, converting a setup.py extension into a publishable .conda, wiring workspace build/publish tasks, or debugging a pixi-build-python sandbox (header-not-found, ModuleNotFoundError, missing sources, sccache misses). Covers the 4-layer config (workspace pixi.toml / package pixi.toml / pyproject.toml / setup.py), CUTLASS-header staging, sm_90a flags, sccache wrappers, and the dev-mode path feature."
user-invocable: true
---

# pixi-build: Package a PyTorch CUDA extension as a conda package

## Core idea

A workspace's CUDA-op subdirectory becomes a self-contained conda package built by `pixi build`. The link works through **four cooperating files**:

| Layer | File | Role |
|---|---|---|
| Workspace orchestration | `pixi.toml` (repo root) | `preview=["pixi-build"]`, build/publish/stage tasks, dev feature path-dep |
| Conda package definition | `<pkg>/pixi.toml` | `[package]`, `pixi-build-python` backend, host/run deps, `extra-input-globs` |
| PEP 517 entrypoint | `<pkg>/pyproject.toml` | `setuptools.build_meta`, project name/version |
| Build implementation | `<pkg>/setup.py` | `CUDAExtension`, include dirs, nvcc flags, sccache wrapper |

The backend `pixi-build-python` reads `pyproject.toml`, lets setuptools produce a wheel, then repackages it into a `.conda`.

## The 5 traps you will hit

These are the failures every first pixi-build setup runs into. Check each.

### 1. Sandbox isolation — files outside `<pkg>/` are invisible
`pixi build --path <pkg>` only sees files **inside the package dir**. CUTLASS headers in `third-party/cutlass/`, sibling Python packages, anything `../`-relative — none of it reaches setup.py. **Stage externals into `<pkg>/external/` via a workspace task before `pixi build`.**

```toml
# root pixi.toml
[tasks.prepare-grouped-mm-sources]
cmd = """
mkdir -p <pkg>/external/cutlass/include &&
mkdir -p <pkg>/external/cutlass/tools/util/include &&
rsync -a --delete third-party/cutlass/include/ <pkg>/external/cutlass/include/ &&
rsync -a --delete third-party/cutlass/tools/util/include/ <pkg>/external/cutlass/tools/util/include/
"""
inputs  = ["third-party/cutlass/include/**", "third-party/cutlass/tools/util/include/**"]
outputs = ["<pkg>/external/cutlass/include/cutlass/cutlass.h"]

[tasks.build-kernels]
cmd = "pixi build --path <pkg> -o $PIXI_PROJECT_ROOT/dist"
depends-on = ["prepare-grouped-mm-sources"]
outputs = ["dist/<pkgname>-*.conda"]
```

Add `<pkg>/external/` to `.gitignore` — it's regenerated.

### 2. `Path(__file__).parent.parent...` math depends on package depth
sst-kernels' `setup.py` at `src/susser_tod/kernels/setup.py` uses `.parent.parent.parent.parent` to reach the workspace root. **Don't blindly copy** — each `.parent` is one `/`. Count your own depth. Even better: don't reach outside the package at all (trap 1).

### 3. setup.py `packages=[...]` must match the actual directory layout
- **Flat layout** (`__init__.py`, `csrc/`, `setup.py` all in `<pkg>/`):
  ```python
  packages=["<import_name>"]
  package_dir={"<import_name>": "."}
  ```
- **Nested layout** (`<pkg>/<import_name>/__init__.py`):
  ```python
  packages=["<import_name>"]
  # no package_dir needed
  ```
Mismatch → empty wheel or `ModuleNotFoundError` at import time, but **build succeeds silently**. Always test with `pixi shell -e dev; python -c "import <import_name>"`.

### 4. Extension name must match `from ._ext import ...`
`CUDAExtension(name="<import_name>._ext", ...)` — the prefix has to be the actual Python package, not a leftover from the template you copied. Otherwise `from ._ext import ...` resolves to a sibling namespace that doesn't exist.

### 5. `extra-input-globs` is for **cache invalidation**, not for including files
It tells pixi "rehash these when they change so the build cache busts." It does NOT pull files into the sandbox. Trap 1 still applies. Always include the staged headers under `<pkg>/external/**` in this list so changes invalidate the cache.

## Reference: minimal correct setup

### `<pkg>/pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "<pkgname>"
version = "0.2.0"
```

### `<pkg>/pixi.toml`
```toml
[package]
name = "<pkgname>"
version = "0.2.0"

[package.build]
backend = { name = "pixi-build-python", version = "*" }

[package.build.config]
extra-input-globs = [
    "csrc/**/*.cu", "csrc/**/*.cuh", "csrc/**/*.cpp", "csrc/**/*.h",
    "external/cutlass/include/**/*.h",
    "external/cutlass/include/**/*.hpp",
    "external/cutlass/include/**/*.cuh",
    "external/cutlass/tools/util/include/**/*.h",
    "external/cutlass/tools/util/include/**/*.hpp",
]
noarch = false
env = { TORCH_CUDA_ARCH_LIST = "9.0", SCCACHE_DIR = "${HOME}/.cache/sccache" }

[package.host-dependencies]   # build-time tools
python = "3.12.*"
setuptools = ">=64"
pytorch = { version = "*", build = "*cuda*" }
cuda = "12.9.*"
pybind11 = "*"
sccache = ">=0.15"

[package.run-dependencies]    # consumer-side runtime deps (baked into metadata)
python = "3.12.*"
pytorch = { version = "*", build = "*cuda*" }
```

### `<pkg>/setup.py` (flat layout, single CUDAExtension)
```python
import os, shutil, sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

PACKAGE_DIR = Path(__file__).parent

# --- sccache wrapping ----------------------------------------------------
# nvcc -ccbin rejects shell-script wrappers, so we only wrap CXX and nvcc
# (not CC). Tiny shell scripts forward to `sccache <real-tool>`.
_sccache = shutil.which("sccache")
if _sccache and not os.environ.get("DISABLE_SCCACHE"):
    _wd = PACKAGE_DIR / ".sccache_wrappers"; _wd.mkdir(exist_ok=True)
    def _wrap(name, target):
        p = _wd / name
        p.write_text(f'#!/bin/sh\nexec {_sccache} {target} "$@"\n')
        p.chmod(0o755); return p
    _real_cxx  = shutil.which(os.environ.get("CXX","g++").split()[-1]) or "g++"
    _real_nvcc = shutil.which("nvcc") or os.path.join(os.environ.get("CUDA_HOME","/usr/local/cuda"),"bin","nvcc")
    os.environ["CXX"] = str(_wrap("cxx", _real_cxx))
    _nvcc_wrap = _wrap("nvcc", _real_nvcc)
    import torch.utils.cpp_extension as _cpp
    _orig = _cpp._join_cuda_home
    _cpp._join_cuda_home = lambda *p: str(_nvcc_wrap) if p == ("bin","nvcc") else _orig(*p)

# --- CUTLASS staged by `pixi run prepare-...-sources` --------------------
cutlass_inc       = PACKAGE_DIR / "external" / "cutlass" / "include"
cutlass_tools_inc = PACKAGE_DIR / "external" / "cutlass" / "tools" / "util" / "include"
if not (cutlass_inc / "cutlass" / "cutlass.h").is_file():
    sys.exit(f"CUTLASS not staged at {cutlass_inc}. Run `pixi run prepare-...-sources`.")

nvcc_arch = [
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-gencode", "arch=compute_90a,code=compute_90a",
    "-D__CUDA_ARCH_FEAT_SM90_ALL",
]

setup(
    name="<pkgname>", version="0.2.0",
    packages=["<import_name>"], package_dir={"<import_name>": "."},
    ext_modules=[CUDAExtension(
        name="<import_name>._ext",
        sources=["csrc/binding.cu"],
        include_dirs=[str(cutlass_inc), str(cutlass_tools_inc)],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3","--use_fast_math","--std=c++17",
                     "--expt-relaxed-constexpr","-ftemplate-backtrace-limit=0",
                     "-Xcompiler=-fno-strict-aliasing","-Xcompiler=-Wconversion"] + nvcc_arch,
        },
    )],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
```

### Root `pixi.toml` — workspace integration
```toml
[workspace]
preview = ["pixi-build"]   # required for `pixi build`
channels = [...]
platforms = ["linux-64"]

[system-requirements]
cuda = "12.9"

[tasks.prepare-<pkg>-sources]   # see Trap 1
...
[tasks.build-kernels]
cmd = "pixi build --path <pkg> -o $PIXI_PROJECT_ROOT/dist"
depends-on = ["prepare-<pkg>-sources"]
outputs = ["dist/<pkgname>-*.conda"]

# Optional publish step
[tasks.publish-packages]
cmd = "publish-packages dist/<pkgname>-*.conda"
depends-on = ["build-kernels"]

# Developers consume the package as a path dep; production installs the .conda
[feature.kernels-dev.dependencies]
<pkgname> = { path = "<pkg>" }

[environments]
dev = ["kernels-dev"]
```

## Build vs dev: two consumption modes

| Mode | How user gets it | When |
|---|---|---|
| **conda binary** | `[dependencies] <pkgname> = "*"` from channel | Production; CI; downstream repos |
| **source path dep** | `pixi shell -e dev` (uses `kernels-dev` feature) | Local development; iterating on kernels |

Path deps recompile on every dependency-graph refresh. Conda binaries skip compilation entirely.

## Verification (always run after wiring)

```bash
# 1. submodule present
git submodule status
ls third-party/cutlass/include/cutlass/cutlass.h

# 2. stage external headers
pixi run prepare-<pkg>-sources
ls <pkg>/external/cutlass/include/cutlass/cutlass.h

# 3. conda build
pixi run build-kernels
ls dist/<pkgname>-*.conda

# 4. dev mode import check (catches Trap 3 / Trap 4)
pixi shell -e dev
python -c "import <import_name>; print(<import_name>.__file__)"

# 5. kernel smoke test
python - <<'PY'
import torch, <import_name>
a = torch.randn(8,128,256, dtype=torch.bfloat16, device='cuda')
b = torch.randn(8,256,512, dtype=torch.bfloat16, device='cuda')
print(<import_name>.grouped_mm(a, b).shape)
PY
```

## Debugging checklist

- **`ModuleNotFoundError: <import_name>._ext`** → trap 3 or 4. Check `packages=[...]` and `CUDAExtension(name=...)` agree with the actual `__init__.py` location.
- **`fatal error: cutlass/cutlass.h: No such file`** → trap 1 or 5. Run the prepare task; verify `<pkg>/external/cutlass/include/cutlass/cutlass.h` exists.
- **Build re-runs every time despite no source changes** → `extra-input-globs` is missing key files. Add them. Also confirm sccache is actually hit: `SCCACHE_LOG=info pixi run build-kernels`.
- **`pixi build` says command not found** → `preview = ["pixi-build"]` missing from root `[workspace]`.
- **Wheel builds but `.conda` is empty / wrong name** → version/name mismatch between `pyproject.toml` and `pixi.toml`. They must agree.
- **`pixi build` works locally but `pixi run build-kernels` doesn't** → `depends-on` not triggering prepare, or `inputs/outputs` are stale. Delete `<pkg>/external/`, re-run.

## What this skill deliberately does not cover

- CUDA kernel correctness, autograd registration, custom-op semantics → covered by general CUDA / PyTorch knowledge.
- Multi-subpackage layout (e.g., sst-kernels' `d2h_scalar/` + `grouped_mm/` side-by-side). Same principles apply, but `packages=[...]` lists each subdir and each gets its own `CUDAExtension`.
- Cross-platform builds — config assumes Linux + sm_90a + CUDA 12.9.
