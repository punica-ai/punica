#!/bin/bash
set -e

assert_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

assert_env PUNICA_CI_PYTHON_VERSION
assert_env PUNICA_CI_TORCH_VERSION
assert_env PUNICA_CI_CUDA_VERSION
assert_env PUNICA_BUILD_VERSION
assert_env TORCH_CUDA_ARCH_LIST
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CONDA_pkgs_dirs=/ci-cache/conda-pkgs
export XDG_CACHE_HOME=/ci-cache/xdg-cache
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
CUDA_MAJOR="${PUNICA_CI_CUDA_VERSION%.*}"
CUDA_MINOR="${PUNICA_CI_CUDA_VERSION#*.}"
PYVER="${PUNICA_CI_PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"


echo "::group::Install PyTorch"
pip install torch==$PUNICA_CI_TORCH_VERSION --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"
echo "::endgroup::"

echo "::group::Install build system"
pip install ninja numpy
pip install --upgrade setuptools wheel build
echo "::endgroup::"


echo "::group::Build wheel for Punica"
cd "$PROJECT_ROOT"
PUNICA_BUILD_VERSION="${PUNICA_BUILD_VERSION}+cu${CUDA_MAJOR}${CUDA_MINOR}" python -m build --no-isolation
rm -f dist/*.tar.gz
python -m build --no-isolation --sdist
echo "::endgroup::"
