"""Two implementations of scaled dot-product attention:
  - naive_attention: materializes the full (seq, seq) score matrix, O(seq^2)
    memory, exactly what Part D asks you to implement first.
  - fused_attention: delegates to torch's fused SDPA, which on a supported
    stack dispatches to a flash-attention or memory-efficient kernel that
    never materializes the full score matrix.
"""
import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_NEW_SDPA_API = True
except ImportError:
    _HAS_NEW_SDPA_API = False


def naive_attention(q, k, v, scale):
    # q, k, v: (batch, heads, seq, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, heads, seq, seq)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def fused_attention(q, k, v, scale):
    if _HAS_NEW_SDPA_API:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)


def make_qkv(batch, heads, seq, head_dim, dtype, device="cuda"):
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    return q, k, v
