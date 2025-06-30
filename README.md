# PyTorch DDP Performance Knobs: Multi-Experiment Project

This repository demonstrates a series of targeted experiments on a small CNN training loop using an implementation of `DistributedDataParallel`. 
Inspired by [Tanishq Kumar's NanoGPT](https://github.com/tanishqkumar/beyond-nanogpt), I wanted to implement Torch DDP and then leverage this module to run some experiments to understand GPU utlization under the hood. Each subfolder isolates one "lever" so you can both run and observe its impact on throughput, memory, and GPU utilization.

## Repository Structure

```
ddp-experiments/
│
├── async_copy/
│   └── train.py
│
├── memory_layout/
│   └── train.py
│
├── precision_fusion/
│   └── train.py
│
├── parallelism/
│   ├── train_ddp.py
│   └── train_pipe.py
│
├── grad_accum/
│   └── train.py
│
├── profiling/
│   ├── train.py
│   └── nsys_capture.sh
│
├── common/
│   ├── model.py         # Shared CNN model definition
│   ├── dataset.py       # Shared dataset utilities
│   └── utils.py         # Any shared helpers (timing, logging, etc.)
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/ddp-experiments.git
   cd ddp-experiments
   ```
2. Setup environment (requires PyTorch with NCCL and CUDA):
    ```bash
    conda create -n ddp-exp python=3.9
    conda activate ddp-exp
    pip install torch torchvision torch-tb-profiler
    ```
3. Run an experiment:

    Each folder contains a train.py with its own README. For example, to test async copies:
    ```bash
    cd async_copy
    torchrun --nproc_per_node=2 train.py --batch-size 1
    ```

## Experiment Details

1. async_copy/

    Goal: Hide Host→GPU memcpy behind kernel execution.

    Uses ``` DataLoader(pin_memory=True)```,```non_blocking=True```, and custom ```torch.cuda.Stream``` examples.

2. memory_layout/

    Goal: Compare default (NCHW) vs. channels-last (NHWC) formats.

    Scripts benchmark convolution throughput with different ```.to(memory_format=…)``` settings.

3. precision_fusion/

    Goal: Evaluate AMP speedup and TorchScript fusion.

    Offers ```--amp``` and ```--jit``` flags to enable autocast and ```torch.jit.script``` respectively.

4. parallelism/

    Goal: Demonstrate DDP vs. pipeline model parallel.

    Two scripts: one wraps model in ```DistributedDataParallel```, the other splits layers across GPUs with ```Pipe```.

5. grad_accum/

    Goal: Show gradient accumulation and delayed sync.

    Implements micro-batch accumulation with ```with model.no_sync()``` and varying accumulation steps.

6. profiling/

    Goal: Collect and visualize performance traces.

    Contains profiler wrappers using ```torch.profiler.profile``` and an ```nsys``` capture script.

## Possible Extensions

- Combine knobs: mix AMP + async copies + NHWC for compounded speedups.
- Swap models/datasets: plug in your own network or real dataset.
- Fine-tune buckets: experiment with different bucket_size in exp_custom_ddp.
- Add NVTX ranges: bracket critical sections to limit nsys capture size.
- Automate metrics: log images/sec, GPU utilization, and memory usage to compare runs.