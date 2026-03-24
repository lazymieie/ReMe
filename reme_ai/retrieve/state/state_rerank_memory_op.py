"""Rerank retrieved state memories with state-aware relevance criteria."""

from flowllm.core.context import C

from reme_ai.retrieve.task.rerank_memory_op import RerankMemoryOp


@C.register_op()
class StateRerankMemoryOp(RerankMemoryOp):
    """Rerank state memories for the current observed state."""

    file_path: str = __file__

    async def async_execute(self):
        if "top_k" in self.context and "top_k" not in self.op_params:
            self.op_params["top_k"] = self.context.top_k
        await super().async_execute()
