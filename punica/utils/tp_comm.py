import os

import torch.distributed as dist


def get_local_rank_from_launcher():
    rank = os.environ.get('LOCAL_RANK')
    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    if rank is None:
        rank = 0
    return int(rank)


def get_world_rank_from_launcher():
    rank = os.environ.get('RANK')
    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_RANK')
    if rank is None:
        rank = 0
    return int(rank)


def get_world_size_from_launcher():
    size = os.environ.get('WORLD_SIZE')
    rank = os.environ.get('RANK')
    if size is None:
        size = os.environ.get('OMPI_COMM_WORLD_SIZE')
    if size is None:
        size = 1
    if rank == 0:
        print(f"set world size to {size}")
    return int(size)


def init_distributed():
    world_rank = get_world_rank_from_launcher()
    world_size = get_world_size_from_launcher()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=world_rank)


def is_distributed():
    return get_world_size_from_launcher() > 1


def create_tensor_parallel_group():
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    ranks = range(world_size)
    mp_group = dist.new_group(ranks)
    return mp_group