#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere. Keeps TripoSR's old transformers/trimesh versions out of
# gaussian-orin, and fixes torchmcubes' CUDA 12 / C++20 lerp name collision.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN=/home/nishantj/miniforge3/bin/conda
TRIPOSR_PYTHON=/home/nishantj/miniforge3/envs/triposr-orin/bin/python
if [[ ! -x "$TRIPOSR_PYTHON" ]]; then
    "$CONDA_BIN" create -y -n triposr-orin --clone gaussian-orin
fi

git -C "$PROJECT_DIR" submodule update --init external_tools/TripoSR
"$TRIPOSR_PYTHON" -m pip install 'numpy<2' 'opencv-python-headless<5' \
    omegaconf==2.3.0 einops==0.7.0 transformers==4.35.0 trimesh==4.0.5 \
    rembg moderngl==5.10.0 xatlas==0.0.11 pybind11 scikit-build-core

if ! "$TRIPOSR_PYTHON" -c 'import torchmcubes' 2>/dev/null; then
    BUILD_DIR="$(mktemp -d /tmp/torchmcubes-orin.XXXXXX)"
    git clone https://github.com/tatsy/torchmcubes.git "$BUILD_DIR/torchmcubes"
    git -C "$BUILD_DIR/torchmcubes" checkout 879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff
    git -C "$BUILD_DIR/torchmcubes" apply "$PROJECT_DIR/patches/torchmcubes-orin-cuda20.patch"
    PYBIND_CMAKE_DIR="$("$TRIPOSR_PYTHON" -m pybind11 --cmakedir)"
    "$CONDA_BIN" run -n triposr-orin env \
        CUDACXX=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=8.7 \
        CMAKE_PREFIX_PATH="$PYBIND_CMAKE_DIR" \
        python -m pip install --no-build-isolation "$BUILD_DIR/torchmcubes"
    echo "Patched torchmcubes source retained at $BUILD_DIR"
fi

"$TRIPOSR_PYTHON" -m pip check
echo "TripoSR environment ready: $TRIPOSR_PYTHON"
