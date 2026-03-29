from nanovllm_labs.base_sequence import BaseSequence
from nanovllm_labs.sampling_params import SamplingParams


class Sequence(BaseSequence):
    def __init__(self, token_ids: list[int], block_size: int, sampling_params: SamplingParams) -> None:
        super().__init__(token_ids, sampling_params)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.block_size = block_size

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]
    
    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        start = index * self.block_size
        end = (index + 1) * self.block_size
        return self.token_ids[start:end]
