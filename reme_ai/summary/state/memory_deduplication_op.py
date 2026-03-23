"""Memory deduplication operation for state memory management."""

from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from loguru import logger

from reme_ai.schema.memory import BaseMemory, StateMemory, vector_node_to_memory


@C.register_op()
class StateMemoryDeduplicationOp(BaseAsyncOp):
    """Remove duplicate state memories against the full state memory pool."""

    file_path: str = __file__

    async def async_execute(self):
        state_memories: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])

        if not state_memories:
            logger.info("No state memories found for deduplication")
            return

        deduplicated_state_memories = await self._deduplicate_state_memories(state_memories)
        logger.info(
            f"Deduplication complete: {len(deduplicated_state_memories)} deduplicated "
            f"state memories out of {len(state_memories)}",
        )
        self.context.response.metadata["memory_list"] = deduplicated_state_memories

    async def _deduplicate_state_memories(self, state_memories: List[BaseMemory]) -> List[BaseMemory]:
        similarity_threshold = self.op_params.get("similarity_threshold", 0.5)
        workspace_id = self.context.get("workspace_id")
        unique_state_memories = []
        existing_embeddings = await self._get_existing_state_memory_embeddings(workspace_id)

        for state_memory in state_memories:
            current_embedding = self._get_state_memory_embedding(state_memory)
            if current_embedding is None:
                logger.warning(f"Failed to generate embedding for state memory: {state_memory.when_to_use[:50]}...")
                continue

            if self._is_similar_to_existing_memories(current_embedding, existing_embeddings, similarity_threshold):
                continue

            if self._is_similar_to_current_batch(current_embedding, unique_state_memories, similarity_threshold):
                continue

            unique_state_memories.append(state_memory)

        return unique_state_memories

    async def _get_existing_state_memory_embeddings(self, workspace_id: str) -> List[List[float]]:
        try:
            if not hasattr(self, "vector_store") or not self.vector_store or not workspace_id:
                return []

            existing_nodes = await self.vector_store.async_search(
                query="...",
                workspace_id=workspace_id,
                top_k=self.op_params.get("max_existing_state_memories", 1000),
            )

            existing_embeddings = []
            for node in existing_nodes:
                parsed_memory = vector_node_to_memory(node)
                if isinstance(parsed_memory, StateMemory) and hasattr(node, "embedding") and node.embedding:
                    existing_embeddings.append(node.embedding)

            logger.debug(
                f"Retrieved {len(existing_embeddings)} existing state memory embeddings from workspace {workspace_id}",
            )
            return existing_embeddings
        except Exception as e:
            logger.warning(f"Failed to retrieve existing state memory embeddings: {e}")
            return []

    def _get_state_memory_embedding(self, state_memory: BaseMemory) -> List[float] | None:
        try:
            text_for_embedding = f"{state_memory.when_to_use} {state_memory.content}"
            embeddings = self.vector_store.embedding_model.get_embeddings([text_for_embedding])
            return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"Error generating embedding for state memory: {e}")
            return None

    def _is_similar_to_existing_memories(
        self,
        current_embedding: List[float],
        existing_embeddings: List[List[float]],
        threshold: float,
    ) -> bool:
        for existing_embedding in existing_embeddings:
            if self._calculate_cosine_similarity(current_embedding, existing_embedding) > threshold:
                return True
        return False

    def _is_similar_to_current_batch(
        self,
        current_embedding: List[float],
        current_state_memories: List[BaseMemory],
        threshold: float,
    ) -> bool:
        for existing_state_memory in current_state_memories:
            existing_embedding = self._get_state_memory_embedding(existing_state_memory)
            if existing_embedding is None:
                continue
            if self._calculate_cosine_similarity(current_embedding, existing_embedding) > threshold:
                return True
        return False

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
