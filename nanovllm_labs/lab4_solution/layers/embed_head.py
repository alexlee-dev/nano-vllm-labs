from nanovllm_labs.common.layers.embed_head import LMHead as CommonLMHead
from nanovllm_labs.common.layers.embed_head import VocabEmbedding as CommonVocabEmbedding
from ..utils.context import get_context


class VocabEmbedding(CommonVocabEmbedding):
    pass


class LMHead(CommonLMHead):
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__(num_embeddings, embedding_dim, get_context, bias=bias)
