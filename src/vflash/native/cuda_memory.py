"""Device capacity available to this allocator, including its unused cache."""


def available_cuda_bytes(torch, device):
    """Count reusable reserved blocks without counting live allocations twice.

    CUDA's free bytes exclude all PyTorch reservations, even inactive blocks
    that the same allocator can reuse. Other processes' memory stays excluded.
    This is an admission estimate; the CUDA allocator still owns allocation.
    """
    free, total = torch.cuda.mem_get_info(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return max(0, min(total, free + reusable))
