import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, PipelineScheduleGPipe
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from parallelism.ddp import DDP

# These environment variables tell PyTorch's distributed backend how to set up communication between processes.
# - MASTER_ADDR: The address (IP or hostname) of the master node (the coordinator for all processes).
#   For single-node jobs, 'localhost' is fine. For multi-node, use the master node's IP.
# - MASTER_PORT: The port on the master node for communication. Can be any free port (e.g., 29500 is common).
# All processes must use the same values for these variables to join the same job.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group(backend='nccl')
rank = dist.get_rank()           # Unique ID for this process (0, 1, ...)
world_size = dist.get_world_size()  # Total number of processes in the job

# Each process gets 2 GPUs: [rank*2, rank*2+1]
# For example, rank 0 gets [0,1], rank 1 gets [2,3].
local_gpus = [rank * 2, rank * 2 + 1]
torch.cuda.set_device(local_gpus[0])  # Set default device for this process

def get_device(idx):
    return torch.device(f'cuda:{idx}')

# The model is split into two stages for pipeline parallelism.
class StandardConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x):
        return self.seq(x)

class StandardLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(5408, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.seq(x)

# Create pipeline stages using the new pipelining API
stage0 = PipelineStage(
    module=StandardConv(),
    stage_index=0,
    num_stages=2,
    device=get_device(local_gpus[0])
)
stage1 = PipelineStage(
    module=StandardLinear(),
    stage_index=1,
    num_stages=2,
    device=get_device(local_gpus[1])
)

# Create a pipeline schedule (GPipe style)
schedule = PipelineScheduleGPipe(
    stages=[stage0, stage1],
    n_microbatches=4,
    loss_fn=nn.CrossEntropyLoss()
)

# Data Preparation (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

# Optimizer for all parameters in both stages
optimizer = optim.SGD(list(stage0.parameters()) + list(stage1.parameters()), lr=0.01)

# Training loop using the new pipelining schedule
for epoch in range(1):
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(local_gpus[0]), target.cuda(local_gpus[0])
        optimizer.zero_grad()
        # This runs forward and backward through the pipeline
        schedule.step(data, target=target)
        optimizer.step()
        if batch_idx % 10 == 0 and rank == 0:
            print(f'Epoch {epoch} Batch {batch_idx} Loss: (see schedule.loss_fn)')

dist.destroy_process_group() 