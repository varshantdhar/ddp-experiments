import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist 
from datasets import load_dataset
from torch.utils.data import DataLoader 
import argparse
from typing import List, Dict 
import os
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from common.model import SimpleCNN
from common.dataset import get_cifar10_loaders

def init(): 
    """
    NCCL (the NVIDIA Collective Communications Library) is the GPU‐optimized backend that PyTorch uses under the hood whenever you specify backend="nccl" in dist.init_process_group.
    In practice, it provides:
    - High-throughput, low-latency collectives like broadcasts, reductions, all-reduce, all-gather, reduce-scatter, barrier, etc.
    - Topology-aware algorithms
        - Automatically discovers your hardware topology (NVLink, PCIe, InfiniBand, etc.) and picks the best ring or tree algorithm to route data
    - Asynchronous GPU-to-GPU transfers by offloading communications onto CUDA streams, NCCL lets you overlap gradient communication with backward computation
    - Multi-node scaling
        - NCCL isn’t just for multi-GPU on one machine—it also works over RDMA or TCP/IP to link GPUs across many machines, scaling your training out to large clusters with minimal extra code
    """
    dist.init_process_group(backend="nccl") # Bootstrapping the distributed process group
    r, wsz = dist.get_rank(), dist.get_world_size()
    # rank is the unique identifier for each process, world_size is the total number of processes 
    torch.cuda.set_device(r)
    # binds each rank to its own GPU (so rank 0 -> cuda:0, rank 1 -> cuda:1, etc.)

    return r, wsz

def shutdown(): 
    dist.destroy_process_group() # tears down the distributed process group

# I know Tanishq's implementation doesn't support grad accumulation, but I'm going to add it here. 
class DDP(nn.Module):
    def __init__(self, model: nn.Module, rank: int, world_size: int, bucket_sz: int = 4): 
        super().__init__()
        self.model = model 
        self.rank = rank 
        self.world_size = world_size 
        self.bucket_sz = bucket_sz
        self.num_buckets = 0
        self.pending_works = []
        self.hooks_to_clean = []

        self._sync_params()  
        buckets, param2bucket = self._make_buckets() # updates self.num_buckets in place 
        self.bucket_counter = [0] * self.num_buckets 
        self.last_bucket_sz = len(buckets[-1])
        self._register_hooks(buckets, param2bucket)

# Argument parsing for flexibility in batch size, epochs, etc.
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP CIFAR-10 Example")
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size per process (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--local_rank', type=int, default=0, help='Local process rank. Provided by torchrun')
    return parser.parse_args()