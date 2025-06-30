# Placeholder for shared dataset utilities 

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size, num_workers=2, pin_memory=True):
    """
    Returns DataLoaders for CIFAR-10 training and test sets.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): How many subprocesses to use for data loading.
            - 0 means data loading is done in the main process.
            - >0 means data loading is done in parallel by that many worker processes.
            - More workers can speed up data loading, especially for large datasets or heavy transforms.
        pin_memory (bool): If True, the DataLoader will copy Tensors into CUDA pinned (page-locked) memory before returning them.
            - Pinned memory enables faster and more efficient transfer of data from host (CPU) to device (GPU) memory.
            - Recommended to set True when using a GPU.
    """
    # Normalize with CIFAR-10 channel means and stds:
    #   mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
    #   These are precomputed over the CIFAR-10 training set and ensure each channel has zero mean and unit variance.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_set, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers, 
                             pin_memory=pin_memory)
    
    return train_loader, test_loader 