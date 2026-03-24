"""Success extraction operation for state memory generation."""

from typing import List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message as FlowMessage
from loguru import logger

from reme_ai.schema import Message, Trajectory
from reme_ai.schema.memory import BaseMemory, StateMemory
from reme_ai.utils.op_utils import get_trajectory_context, merge_messages_content, parse_json_experience_response

DEFAULT_STATE_NAME = "default_state"


@C.register_op()
class StateSuccessExtractionOp(BaseAsyncOp):
    """Extract successful state-level experiences from trajectories."""

    file_path: str = __file__

    async def async_execute(self):
        success_trajectories: List[Trajectory] = self.context.get("success_trajectories", [])

        if not success_trajectories:
            logger.info("No success trajectories found for state extraction")
            return

        success_state_memories = []
        for trajectory in success_trajectories:
            segments = trajectory.metadata.get("segments", [])
            if segments:
                for segment in segments:
                    success_state_memories.extend(await self._extract_from_steps(segment, trajectory))
            else:
                success_state_memories.extend(await self._extract_from_steps(trajectory.messages, trajectory))

        self.context.success_state_memories = success_state_memories
        logger.info(f"Extracted {len(success_state_memories)} success state memories")

    async def _extract_from_steps(self, steps: List[Message], trajectory: Trajectory) -> List[BaseMemory]:
        state_name = trajectory.metadata.get("state_name", self.context.get("state_name", DEFAULT_STATE_NAME))
        logger.info(
            "StateSuccessExtractionOp using llm model={} base_url={} state_name={} step_count={}",
            getattr(self.llm, "model_name", ""),
            getattr(self.llm, "base_url", ""),
            state_name,
            len(steps),
        )
        prompt = self.prompt_format(
            prompt_name="success_step_state_memory_prompt",
            query=trajectory.metadata.get("query", ""),
            state_name=state_name,
            step_sequence=merge_messages_content(steps),
            context=get_trajectory_context(trajectory, steps),
            outcome="successful",
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
