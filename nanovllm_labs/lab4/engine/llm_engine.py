from nanovllm_labs.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        pass

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        raise NotImplementedError

    def exit(self) -> None: ...
    