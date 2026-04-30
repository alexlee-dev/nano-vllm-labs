from nanovllm_labs.common.layers.attention import PagedKVAttention
from ..utils.context import get_context


class Attention(PagedKVAttention):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__(num_heads, head_dim, scale, num_kv_heads, get_context)
