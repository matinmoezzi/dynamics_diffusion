import os
import socket
import torch
from mpi4py import MPI
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size, hostname):
    print(f"I am {rank} of {size} in {hostname}")
    tensor = torch.zeros(1)
    device = torch.device("cuda:{}".format(rank))
    tensor = tensor.to(device)
    if rank == 0:
        for rank_recv in range(1, size):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(rank, 0))



def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", world_size=comm.size, rank=comm.rank, init_method="env://")
    run(comm.rank, comm.size, hostname)