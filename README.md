# Flash-Attention-3 Forward-Only Kernel (lishunyang12 fork)

> **Patched fork notice:** This fork of [ZJY0516/fa-fwd](https://github.com/ZJY0516/fa-fwd) pre-applies the FP8 two-level accumulation patch from [vllm-project/flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104) on top of the upstream `flash-attention/hopper/` sources. The Python package name is kept as `fa3_fwd` so downstream code (e.g. `vllm-omni`) can use `from fa3_fwd_interface import ...` with no changes. The patch can be disabled at build time by exporting `FLASH_ATTENTION_DISABLE_FP8_TWO_LEVEL_ACCUMULATION=TRUE`.

## Install from a release

Pre-built wheels are published as GitHub Release assets on this fork. To install directly:

```bash
pip install https://github.com/lishunyang12/fa-fwd/releases/download/v0.0.2-lsy1/fa3_fwd-0.0.2-cp39-abi3-linux_x86_64.whl \
    --force-reinstall --no-deps
```

Replace the tag (`v0.0.2-lsy1`) and wheel filename with whichever release you want. `--no-deps` avoids pulling in PyTorch; make sure your environment already has a matching CUDA-enabled PyTorch.

## Triggering a new build

The GitHub Actions workflow at [.github/workflows/build-wheel.yml](.github/workflows/build-wheel.yml) builds a Hopper (sm_90a) wheel automatically. Two ways to trigger it:

1. **Push a version tag** — this builds and creates a GitHub Release with the wheel attached:
   ```bash
   git tag v0.0.2-lsy2
   git push origin v0.0.2-lsy2
   ```
2. **Manual dispatch** — run the `Build Wheel` workflow from the Actions tab; the wheel is uploaded as a build artifact.

CUDA wheel builds on the free GitHub runners take **1–2 hours** and use `MAX_JOBS=2 NVCC_THREADS=1` to stay within the 7 GB runner memory limit.

### Alternative: upload a pre-built wheel manually

If you already have a wheel built on your own machine, you can attach it to a release directly:

```bash
gh release upload v0.0.2-lsy1 fa3_fwd-0.0.2-cp39-abi3-linux_x86_64.whl \
    --repo lishunyang12/fa-fwd
```

---

This repository bundles the Flash-Attention-3 forward-only kernel and the tooling required to build a lightweight Python wheel. It is intended for inference scenarios where backward operators and optional features are unnecessary.

## Highlights
- Ships only the Flash-Attention-3 forward path while disabling backward kernels, local attention, paged KV cache, FP16 kernels, and other extras to minimize the wheel size.
- Applies a patch that renames the public interface to `fa3_fwd_interface`, making the forward kernel easy to import from Python.

## Prerequisites(same as upstream)
- **Python**: 3.9 or later
- **PyTorch**: 2.10
- **Build dependencies**: `ninja`, `packaging`, `wheel`

## Quick Start
1. Clone the repository and initialize submodules:
	```bash
	git clone --recursive <repo-url>
	cd fa3-fwd
	# If --recursive was omitted during clone, run:
	git submodule update --init --recursive
	```
2. Create a Python virtual environment and install dependencies:
	```bash
	uv venv --python 3.12 --seed
	source .venv/bin/activate
	uv pip install -r requirements.txt
	```
3. Build the forward-only wheel:
	```bash
	bash build_fa3.sh
	```
	The script:
	- Sources [set_compile_env.sh](set_compile_env.sh) to compute `MAX_JOBS` and `NVCC_THREADS`
	- Applies the custom patch and interface rename inside the Flash-Attention submodule
	- Runs `python setup.py bdist_wheel` under [flash-attention/hopper](flash-attention/hopper)

4. Install the generated wheel (example):
	```bash
	pip install build/*.whl
	```

## Python Usage Example
```python
import torch
from fa3_fwd_interface import flash_attn_func

# Inputs must already live on CUDA and satisfy Flash-Attention-3 constraints
out = flash_attn_func(q, k, v, causal=True)
```

> This package exposes only the forward kernel. For backward support or additional features, depend on the upstream Flash-Attention project instead.


## Troubleshooting
- **Out-of-memory during compilation**: The build script already throttles concurrency, but you can enforce `MAX_JOBS=1 NVCC_THREADS=1` before running `bash build_fa3.sh`.
- **CUDA mismatch errors**: Confirm that `nvcc --version` aligns with `torch.version.cuda`.

## Repository Layout
- [build_fa3.sh](build_fa3.sh): Main build entry point
- [set_compile_env.sh](set_compile_env.sh): Resource-based compiler configuration helper
- [hopper_setup_py.patch](hopper_setup_py.patch): Patch applied to the upstream `setup.py` (package rename, long-description path fix)
- [two_level_accum.patch](two_level_accum.patch): FP8 two-level accumulation patch from [vllm-project/flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104)
- [.github/workflows/build-wheel.yml](.github/workflows/build-wheel.yml): CI workflow that builds and releases wheels on tag push
- [flash-attention](flash-attention): Upstream Flash-Attention submodule

Customize further by editing environment variables in the build script or modifying the submodule before the patch is applied (for example to re-enable additional datatypes or kernels).
