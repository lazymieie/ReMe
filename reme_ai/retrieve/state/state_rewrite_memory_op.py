"""Rewrite retrieved state memories into state-aware guidance."""

from typing import List

from flowllm.core.context import C

from reme_ai.retrieve.task.rewrite_memory_op import RewriteMemoryOp
from reme_ai.schema.memory import BaseMemory


@C.register_op()
class StateRewriteMemoryOp(RewriteMemoryOp):
    """Rewrite retrieved state memories into guidance for the current state."""

    file_path: str = __file__

    async def async_execute(self):
        memory_list = self.context.response.metadata.get("memory_list", [])
        if not memory_list:
            if not self.context.response.answer:
                self.context.response.answer = "No matching state memories found"
            return
        await super().async_execute()

    @staticmethod
    def _format_memories_for_context(memories: List[BaseMemory]) -> str:
        formatted_memories = []

        for i, memory in enumerate(memories, 1):
            condition = memory.when_to_use or "(no explicit trigger recorded)"
            content = memory.content
            memory_text = (
                f"State Memory {i}:\n"
                f"Observed State / Trigger: {condition}\n"
                f"Implication / Recommended Action: {content}\n"
            )
            formatted_memories.append(memory_text)

        return "\n".join(formatted_memories)
