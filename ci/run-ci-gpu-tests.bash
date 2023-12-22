#!/bin/bash
set -e

assert_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

assert_env PUNICA_CI_TORCH_VERSION
assert_env PUNICA_CI_CUDA_MAJOR
assert_env PUNICA_CI_CUDA_MINOR
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CONDA_pkgs_dirs=/ci-cache/conda-pkgs
export XDG_CACHE_HOME=/ci-cache/xdg-cache
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
nvidia-smi


echo "::group::Install Mamba and Python"
if [ ! -f "/ci-cache/Miniforge3.sh" ]; then
    wget -O "/ci-cache/Miniforge3.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
fi
bash "/ci-cache/Miniforge3.sh" -b -p "$HOME/conda"
source "$HOME/conda/etc/profile.d/conda.sh"
source "$HOME/conda/etc/profile.d/mamba.sh"
mamba create -y -n punica-ci python=3.10.13 git
mamba activate punica-ci
echo "::endgroup::"


echo "::group::Install PyTorch"
pip install torch==$PUNICA_CI_TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${PUNICA_CI_CUDA_MAJOR}${PUNICA_CI_CUDA_MINOR}"
echo "::endgroup::"


echo "::group::Install Punica"
cd "$PROJECT_ROOT"
pip install ninja numpy
pip install -v --no-build-isolation -e .[dev]
echo "::endgroup::"


echo "::group::Punica pytest"
pytest --cov=src --cov-report=xml -v
echo "::endgroup::"
