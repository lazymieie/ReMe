"""State memory retrieval operation module."""

from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import VectorNode
from loguru import logger

from reme_ai.schema.memory import StateMemory, vector_node_to_memory

DEFAULT_STATE_NAME = "default_state"


@C.register_op()
class RetrieveStateMemoryOp(BaseAsyncOp):
    """Retrieve state memories by state_name and query."""

    file_path: str = __file__

    @staticmethod
    def _format_state_memories(memories: List[StateMemory]) -> str:
        lines = [f"Retrieved {len(memories)} state memory(ies):\n"]

        for idx, memory in enumerate(memories, 1):
            lines.append(f"State Target: {memory.target}")
            lines.append(f"When to use: {memory.when_to_use}")
            lines.append(f"Content: {memory.content}")

            if idx < len(memories):
                lines.append("\n---\n")

        return "\n".join(lines)

    async def async_execute(self):
        query: str = self.context.get("query", "")
        state_name: str = self.context.get("state_name", "") or DEFAULT_STATE_NAME
        workspace_id: str = self.context.workspace_id
        top_k: int = self.context.get("top_k", 5)

        if not query:
            logger.warning("query is empty, skipping processing")
            self.context.response.answer = "query is required"
            self.context.response.success = False
            return

        logger.info(f"workspace_id={workspace_id} retrieving state memory for target={state_name}, top_k={top_k}")

        nodes: List[VectorNode] = await self.vector_store.async_search(
            query=query,
            workspace_id=workspace_id,
            top_k=max(top_k * 3, top_k),
        )

        matched_state_memories: List[StateMemory] = []
        seen_contents: set[str] = set()
        for node in nodes:
            memory = vector_node_to_memory(node)
            if (
                isinstance(memory, StateMemory)
                and memory.target == state_name
                and memory.content not in seen_contents
            ):
                matched_state_memories.append(memory)
                seen_contents.add(memory.content)
                if len(matched_state_memories) >= top_k:
                    break

        if not matched_state_memories:
            logger.info("No matching state memories found")
            self.context.response.answer = "No matching state memories found"
            self.context.response.success = False
            return

        self.context.response.answer = self._format_state_memories(matched_state_memories)
        self.context.response.success = True
        self.context.response.metadata["memory_list"] = matched_state_memories

        for memory in matched_state_memories:
            logger.info(
                f"Retrieved state memory: target={memory.target}, when_to_use={memory.when_to_use}",
            )
