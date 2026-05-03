from nanovllm_labs.common.context import Context, ContextStore

_STORE = ContextStore(thread_local=True)
get_context = _STORE.get_context
set_context = _STORE.set_context
reset_context = _STORE.reset_context

__all__ = ["Context", "get_context", "set_context", "reset_context"]
