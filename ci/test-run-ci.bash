docker run --runtime=nvidia --gpus all --rm -t \
    -v $HOME/ci-test/cache:/ci-cache \
    -v $HOME/ci-test/punica-checkout:/app \
    -e PUNICA_CI_TORCH_VERSION=2.1.0 \
    -e PUNICA_CI_CUDA_MAJOR=12 \
    -e PUNICA_CI_CUDA_MINOR=1 \
    --user $(id -u):$(id -g) \
    nvidia/cuda:12.1.0-devel-ubuntu22.04 \
    bash /app/ci/run-ci-gpu-tests.bash
