import torch
import torch.distributed as dist

def reduce_tensor(tensor):
    if torch.is_tensor(tensor):
        rt = tensor.clone()
    else:
        rt = torch.Tensor([tensor]).cuda()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt