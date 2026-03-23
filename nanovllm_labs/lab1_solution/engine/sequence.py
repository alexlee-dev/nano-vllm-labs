from nanovllm_labs.base_sequence import BaseSequence


class Sequence(BaseSequence):
    """Lab 1 request state.

    BaseSequence already stores everything needed for this lab:
    prompt tokens, generated tokens, and sampling-related fields.

    The subclass exists mainly for two reasons:
    1. It gives the lab a concrete request type to pass around.
    2. It leaves a clean place to add lab-specific state later.
    """

    pass
