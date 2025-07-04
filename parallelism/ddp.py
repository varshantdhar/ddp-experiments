import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager

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
        buckets, param2bucket = self._make_buckets()  # updates self.num_buckets in place
        self.bucket_counter = [0] * self.num_buckets
        self.last_bucket_sz = len(buckets[-1])
        self._register_hooks(buckets, param2bucket)

    def _sync_params(self):
        """
        Broadcast all model parameters from rank 0 to all other ranks.
        Ensures all processes start with identical model weights.
        """
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def _make_buckets(self):
        """
        Groups model parameters into buckets of size self.bucket_sz.
        Returns:
            buckets (List[List[Parameter]]): List of parameter buckets.
            param2bucket (Dict[Parameter, int]): Mapping from parameter to its bucket index.
        The last bucket may be smaller if the total number of parameters is not a multiple of bucket_sz.
        This 'leftover bucket' ensures all parameters are included in gradient synchronization.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        buckets = [params[i:i+self.bucket_sz] for i in range(0, len(params), self.bucket_sz)]
        param2bucket = {p: i for i, bucket in enumerate(buckets) for p in bucket}
        self.num_buckets = len(buckets)
        return buckets, param2bucket

    def _register_hooks(self, buckets, param2bucket):
        """
        Registers backward hooks on each parameter to mark gradients as ready for synchronization.
        When all grads in a bucket are ready, triggers an asynchronous all-reduce for that bucket's gradients.
        All pending work handles are stored for later synchronization.
        """
        self.bucket_ready = [0] * len(buckets)
        self.buckets = buckets  # Save for synchronize()
        self.pending_works = []  # Store async work handles

        def make_hook(param, bucket_idx):
            def hook(grad):
                self.bucket_ready[bucket_idx] += 1
                if self.bucket_ready[bucket_idx] == len(buckets[bucket_idx]):
                    # Only perform all-reduce if not in no_sync() context; otherwise, skip to accumulate gradients locally
                    if not getattr(self, '_no_sync', False):
                        grads = [p.grad for p in buckets[bucket_idx] if p.grad is not None]
                        for g in grads:
                            work = dist.all_reduce(g, async_op=True)
                            self.pending_works.append(work)
                    # Reset counter for next accumulation cycle
                    self.bucket_ready[bucket_idx] = 0
            return hook

        # Register the hook for each parameter
        # Note: These hooks are registered on the underlying model's parameters.
        # When you call loss.backward(), PyTorch's autograd engine will automatically call these hooks
        # as soon as the gradient for each parameter is computed, even though this is a custom nn.Module subclass.
        for p in self.model.parameters():
            if p.requires_grad:
                bucket_idx = param2bucket[p]
                h = p.register_hook(make_hook(p, bucket_idx))
                self.hooks_to_clean.append(h)

    def synchronize(self):
        """
        Waits for all asynchronous all-reduce operations to finish.
        This should be called after gradient accumulation steps, before optimizer.step().
        """
        for w in self.pending_works:
            w.wait()
        self.pending_works = []

    def forward(self, *args, **kwargs):
        """
        Forward pass: delegates to the wrapped model.
        Allows this module to be used like a standard PyTorch model.
        """
        return self.model(*args, **kwargs)

    def __del__(self):
        """
        Cleanup: ensure all async all-reduce ops are finished and remove all registered hooks.
        Prevents memory leaks, dangling references, and unfinished communication.
        """
        if hasattr(self, 'synchronize'):
            try:
                self.synchronize()
            except Exception:
                pass  # Avoid raising exceptions in __del__
        for h in getattr(self, 'hooks_to_clean', []):
            h.remove()

    @contextmanager
    def no_sync(self):
        """
        Context manager to temporarily disable gradient synchronization (all-reduce) during backward passes.
        Useful for gradient accumulation: gradients are accumulated locally until sync is re-enabled.
        """
        if not hasattr(self, '_no_sync'):  # Initialize flag if not present
            self._no_sync = False
        old_flag = self._no_sync
        self._no_sync = True
        try:
            yield
        finally:
            self._no_sync = old_flag 