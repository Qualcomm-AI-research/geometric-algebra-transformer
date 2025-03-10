class AttentionBias:
    pass

def memory_efficient_attention(*args, **kwargs):
    import torch
    import torch.nn.functional as F
    q, k, v = args[:3]
    # Fall back to standard PyTorch attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
