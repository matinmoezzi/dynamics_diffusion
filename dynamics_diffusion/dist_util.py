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
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

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
            return th.device(f"cpu:{os.environ['LOCAL_RANK']}")
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
