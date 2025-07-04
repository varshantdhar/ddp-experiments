import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist 
from datasets import load_dataset
from torch.utils.data import DataLoader 
import argparse
from typing import List, Dict 
import os
from contextlib import contextmanager
from parallelism.ddp import DDP

def init(): 
    """
    NCCL (the NVIDIA Collective Communications Library) is the GPU‐optimized backend that PyTorch uses under the hood whenever you specify backend="nccl" in dist.init_process_group.
    In practice, it provides:
    - High-throughput, low-latency collectives like broadcasts, reductions, all-reduce, all-gather, reduce-scatter, barrier, etc.
    - Topology-aware algorithms
        - Automatically discovers your hardware topology (NVLink, PCIe, InfiniBand, etc.) and picks the best ring or tree algorithm to route data
    - Asynchronous GPU-to-GPU transfers by offloading communications onto CUDA streams, NCCL lets you overlap gradient communication with backward computation
    - Multi-node scaling
        - NCCL isn't just for multi-GPU on one machine—it also works over RDMA or TCP/IP to link GPUs across many machines, scaling your training out to large clusters with minimal extra code
    """
    dist.init_process_group(backend="nccl") # Bootstrapping the distributed process group
    r, wsz = dist.get_rank(), dist.get_world_size()
    # rank is the unique identifier for each process, world_size is the total number of processes 
    torch.cuda.set_device(r)
    # binds each rank to its own GPU (so rank 0 -> cuda:0, rank 1 -> cuda:1, etc.)

    return r, wsz

def shutdown(): 
    dist.destroy_process_group() # tears down the distributed process group

# Argument parsing for flexibility in batch size, epochs, etc.
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP CIFAR-10 Example")
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size per process (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--local_rank', type=int, default=0, help='Local process rank. Provided by torchrun')
    return parser.parse_args()

def main():
    args = parse_args()
    r, wsz = init()
    device = torch.device(f'cuda:{r}')
    # Each process runs this script independently. We need to initialize the process group so they can communicate.
    # This sets up the backend (NCCL for GPUs) and connects all processes in the job.
    dist.init_process_group(backend="nccl")

    # Each process should use only its assigned GPU. local_rank is provided by torchrun.
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # Example: Minimal training loop using custom DDP (no gradient accumulation)
    from common.model import SimpleCNN
    from common.dataset import get_cifar10_loaders
    import torch.optim as optim
    import torch.nn as nn
    import torch.distributed as dist
    import torch

    # Set up model and wrap with custom DDP
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, rank=r, world_size=wsz)

    # Set up data loader with DistributedSampler
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_loader.dataset, num_replicas=wsz, rank=r, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=args.batch_size, sampler=train_sampler)

    # Set up optimizer and loss
    optimizer = optim.SGD(ddp_model.model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            ddp_model.synchronize()  # Ensure all async all-reduce ops finish
            optimizer.step()
            if batch_idx % 10 == 0 and r == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()