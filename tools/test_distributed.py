import torch
import torch.distributed as dist
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch Distributed Test')
parser.add_argument('--dist-backend',
                    default='gloo',
                    type=str,
                    help='distributed backend')


def run_allreduce(rank, size, n=10):
    """use cuda tensor if backend is `nccl`
    """
    data = torch.randn(size)
    t = time.time()
    for _ in range(n):
        dist.all_reduce(data, op=dist.reduce_op.SUM)
    print('average time:', (time.time() - t) / n)


if __name__ == '__main__':
    addr = '127.0.0.1'
    addr = 'tcp://' + addr + ':23458'
    print(addr)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parser.parse_args()
    dist.init_process_group(backend=args.dist_backend,
                            init_method=addr,
                            world_size=size,
                            rank=rank)

    run_allreduce(rank, 1024 * 1024 * 25)
