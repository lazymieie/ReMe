"""Comparative extraction operation for state memory generation."""

from typing import List, Optional, Tuple

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message as FlowMessage
from loguru import logger

from reme_ai.schema import Message, Trajectory
from reme_ai.schema.memory import BaseMemory, StateMemory
from reme_ai.utils.op_utils import merge_messages_content, parse_json_experience_response

DEFAULT_STATE_NAME = "default_state"


@C.register_op()
class StateComparativeExtractionOp(BaseAsyncOp):
    """Extract comparative state memories across trajectories."""

    file_path: str = __file__

    async def async_execute(self):
        all_trajectories: List[Trajectory] = self.context.get("all_trajectories", [])
        success_trajectories: List[Trajectory] = self.context.get("success_trajectories", [])
        failure_trajectories: List[Trajectory] = self.context.get("failure_trajectories", [])

        comparative_state_memories = []

        if len(all_trajectories) >= 2 and self.op_params.get("enable_soft_comparison", True):
            highest_traj, lowest_traj = self._find_highest_lowest_scoring_trajectories(all_trajectories)
            if highest_traj and lowest_traj and highest_traj.score > lowest_traj.score:
                comparative_state_memories.extend(
                    await self._extract_soft_comparative_state_memory(highest_traj, lowest_traj),
                )

        if success_trajectories and failure_trajectories and self.op_params.get("enable_similarity_comparison", False):
            similar_pairs = self._find_similar_step_sequences(success_trajectories, failure_trajectories)
            logger.info(f"Found {len(similar_pairs)} similar state pairs for hard comparison")
            for success_steps, failure_steps, similarity_score, state_name in similar_pairs:
                comparative_state_memories.extend(
                    await self._extract_hard_comparative_state_memory(
                        success_steps,
                        failure_steps,
                        similarity_score,
                        state_name,
                    ),
                )

        self.context.comparative_state_memories = comparative_state_memories
        logger.info(f"Extracted {len(comparative_state_memories)} comparative state memories")

    @staticmethod
    def _find_highest_lowest_scoring_trajectories(
        trajectories: List[Trajectory],
    ) -> Tuple[Optional[Trajectory], Optional[Trajectory]]:
        if len(trajectories) < 2:
            return None, None

        valid_trajectories = [traj for traj in trajectories if traj.score is not None]
        if len(valid_trajectories) < 2:
            logger.warning("Not enough trajectories with valid scores for state comparison")
            return None, None

        sorted_trajectories = sorted(valid_trajectories, key=lambda x: x.score, reverse=True)
        return sorted_trajectories[0], sorted_trajectories[-1]

    async def _extract_soft_comparative_state_memory(
        self,
        higher_traj: Trajectory,
        lower_traj: Trajectory,
    ) -> List[BaseMemory]:
        state_name = higher_traj.metadata.get("state_name", self.context.get("state_name", DEFAULT_STATE_NAME))
        prompt = self.prompt_format(
            prompt_name="soft_comparative_step_state_memory_prompt",
            state_name=state_name,
            higher_steps=merge_messages_content(self._get_trajectory_steps(higher_traj)),
            lower_steps=merge_messages_content(self._get_trajectory_steps(lower_traj)),
            higher_score=f"{higher_traj.score:.2f}",
            lower_score=f"{lower_traj.score:.2f}",
        )

        def parse_state_memories(message: Message) -> List[BaseMemory]:
            return self._build_state_memories(message.content, state_name)

        return await self.llm.achat(
            messages=[FlowMessage(role=Role.USER, content=prompt)],
            callback_fn=parse_state_memories,
        )

    async def _extract_hard_comparative_state_memory(
        self,
        success_steps: List[Message],
        failure_steps: List[Message],
        similarity_score: float,
        state_name: str,
    ) -> List[BaseMemory]:
        prompt = self.prompt_format(
            prompt_name="hard_comparative_step_state_memory_prompt",
            state_name=state_name,
            success_steps=merge_messages_content(success_steps),
            failure_steps=merge_messages_content(failure_steps),
            similarity_score=similarity_score,
        )

        def parse_state_memories(message: Message) -> List[BaseMemory]:
            return self._build_state_memories(message.content, state_name)

        return await self.llm.achat(
            messages=[FlowMessage(role=Role.USER, content=prompt)],
            callback_fn=parse_state_memories,
        )

    def _build_state_memories(self, content: str, state_name: str) -> List[BaseMemory]:
        memories_data = parse_json_experience_response(content)
        state_memories = []

        for memory_data in memories_data:
            when_to_use = memory_data.get("when_to_use", memory_data.get("condition", "")).strip()
            experience = memory_data.get("experience", "").strip()
            if when_to_use and experience:
                state_memories.append(
                    StateMemory(
                        workspace_id=self.context.get("workspace_id", ""),
                        when_to_use=when_to_use,
                        content=experience,
                        target=state_name,
                        author=getattr(self.llm, "model_name", "system"),
                        metadata=memory_data,
                    ),
                )

        return state_memories

    @staticmethod
    def _get_trajectory_steps(trajectory: Trajectory) -> List[Message]:
        segments = trajectory.metadata.get("segments", [])
        if segments:
            all_steps = []
            for segment in segments:
                all_steps.extend(segment)
            return all_steps
        return trajectory.messages

    def _find_similar_step_sequences(
        self,
        success_trajectories: List[Trajectory],
        failure_trajectories: List[Trajectory],
    ) -> List[Tuple[List[Message], List[Message], float, str]]:
        if not self.op_params.get("enable_similarity_comparison", False):
            return []

        try:
            success_step_sequences = []
            for traj in success_trajectories:
                state_name = traj.metadata.get("state_name", self.context.get("state_name", DEFAULT_STATE_NAME))
                segments = traj.metadata.get("segments", [])
                if segments:
                    success_step_sequences.extend((segment, state_name) for segment in segments)
                else:
                    success_step_sequences.append((traj.messages, state_name))

            failure_step_sequences = []
            for traj in failure_trajectories:
                state_name = traj.metadata.get("state_name", self.context.get("state_name", DEFAULT_STATE_NAME))
                segments = traj.metadata.get("segments", [])
                if segments:
                    failure_step_sequences.extend((segment, state_name) for segment in segments)
                else:
                    failure_step_sequences.append((traj.messages, state_name))

            max_sequences = self.op_params.get("max_similarity_sequences", 5)
            success_step_sequences = success_step_sequences[:max_sequences]
            failure_step_sequences = failure_step_sequences[:max_sequences]

            if not success_step_sequences or not failure_step_sequences:
                return []

            success_texts = [merge_messages_content(seq) for seq, _ in success_step_sequences]
            failure_texts = [merge_messages_content(seq) for seq, _ in failure_step_sequences]

            if hasattr(self, "vector_store") and self.vector_store and hasattr(self.vector_store, "embedding_model"):
                success_embeddings = self.vector_store.embedding_model.get_embeddings(success_texts)
                failure_embeddings = self.vector_store.embedding_model.get_embeddings(failure_texts)

                similar_pairs = []
                similarity_threshold = self.op_params.get("similarity_threshold", 0.3)

                for i, s_emb in enumerate(success_embeddings):
                    for j, f_emb in enumerate(failure_embeddings):
                        similarity = self._calculate_cosine_similarity(s_emb, f_emb)
                        if similarity > similarity_threshold:
                            success_steps, state_name = success_step_sequences[i]
                            failure_steps, _ = failure_step_sequences[j]
                            similar_pairs.append((success_steps, failure_steps, similarity, state_name))

                max_pairs = self.op_params.get("max_similarity_pairs", 3)
                return sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:max_pairs]
        except Exception as e:
            logger.error(f"Error finding similar state step sequences: {e}")

        return []

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
