#!/bin/bash
set -e

assert_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

echo "::group::Environment"
assert_env PUNICA_CI_CACHE_DIR
assert_env PUNICA_CI_TORCH_VERSION
assert_env PUNICA_CI_CUDA_MAJOR
assert_env PUNICA_CI_CUDA_MINOR
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CONDA_pkgs_dirs="$PUNICA_CI_CACHE_DIR/conda-pkgs"
PIP_CACHE_DIR="$PUNICA_CI_CACHE_DIR/pip"
mkdir -p "$CONDA_pkgs_dirs" "$PIP_CACHE_DIR"
nvidia-smi
echo "::endgroup::"


echo "::group::Install Mamba and Python"
wget -O "$PUNICA_CI_CACHE_DIR/Miniforge3.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
"$PUNICA_CI_CACHE_DIR/Miniforge3.sh" -b -p "/tmp/conda"
source "/tmp/conda/etc/profile.d/conda.sh"
source "/tmp/conda/etc/profile.d/mamba.sh"
mamba create -y -n punica-ci python=3.10
mamba activate punica-ci
echo "::endgroup::"


echo "::group::Install PyTorch"
pip install torch==$PUNICA_CI_TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${PUNICA_CI_CUDA_MAJOR}${PUNICA_CI_CUDA_MINOR}"
echo "::endgroup::"


echo "::group::Install Punica"
cd "$PROJECT_ROOT"
pip install ninja numpy
pip install -v --no-build-isolation .[dev]
echo "::endgroup::"


echo "::group::Punica pytest"
coverage run -m pytest -v
echo "::endgroup::"
