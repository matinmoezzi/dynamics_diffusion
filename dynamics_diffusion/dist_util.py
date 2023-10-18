"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def find_free_port():
    """
    Finds a free port on the host.

    Returns:
        int: A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(
            ("localhost", 0)
        )  # Bind to port 0, which prompts the OS to assign a free port
        s.listen(1)  # Start listening for connections on the socket
        port = s.getsockname()[1]  # Retrieve the port number assigned by the OS
        return port


class DistUtil:
    _instance = None
    device = "cude"

    @classmethod
    def setup_dist(cls, device="cuda"):
        """
        Setup a distributed process group.
        """
        if dist.is_initialized():
            return
        if device:
            cls.device = device

        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_port()))

        backend = "gloo" if device == "cpu" else "nccl"

        dist.init_process_group(backend=backend)
        if device == "cuda":
            th.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        if cls._instance is None:
            cls._instance = cls()

    @classmethod
    def dev(cls):
        """
        Get the device to use for torch.distributed.
        """
        if cls.device == "cpu":
            return th.device(f"cpu")
        else:
            if th.cuda.is_available():
                if dist.is_initialized():
                    return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
                return th.device(f"cuda")
            return th.device("cpu")

    @classmethod
    def get_local_rank(cls):
        """
        Get the local rank of the current process.
        """
        if not dist.is_initialized():
            return 0
        return int(os.environ["LOCAL_RANK"])

    @classmethod
    def get_global_rank(cls):
        """
        Get the local rank of the current process.
        """
        if not dist.is_initialized():
            return 0
        return int(os.environ["RANK"])
