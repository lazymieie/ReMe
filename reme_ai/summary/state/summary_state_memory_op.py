"""State memory summary operation."""

import json
from typing import List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message as FlowMessage
from loguru import logger

from reme_ai.schema import Message
from reme_ai.schema.memory import BaseMemory, StateMemory
from reme_ai.utils.op_utils import merge_messages_content

DEFAULT_STATE_NAME = "default_state"


@C.register_op()
class SummaryStateMemoryOp(BaseAsyncOp):
    """Summarize message trajectories into reusable state memories."""

    file_path: str = __file__

    async def async_execute(self):
        messages: list = self.context.get("messages", [])
        state_name: str = self.context.get("state_name", "") or DEFAULT_STATE_NAME

        if not messages:
            self.context.response.answer = "messages is required"
            self.context.response.success = False
            return

        normalized_messages: List[Message] = [Message(**x) if isinstance(x, dict) else x for x in messages]
        memory_list = await self.summary_messages(normalized_messages, state_name)

        self.context.response.answer = json.dumps([x.model_dump() for x in memory_list], ensure_ascii=False)
        self.context.response.metadata["memory_list"] = memory_list
        for memory in memory_list:
            logger.info(
                f"add state memory: target={memory.target} when_to_use={memory.when_to_use}\ncontent={memory.content}",
            )

    async def summary_messages(self, messages: List[Message], state_name: str) -> List[BaseMemory]:
        """Extract state memories from messages."""
        execution_process = merge_messages_content(messages)
        summary_prompt = self.prompt_format(
            prompt_name="summary_prompt",
            execution_process=execution_process,
            state_name=state_name,
            summary_example=self.get_prompt("summary_example"),
        )

        def parse_content(message: Message):
            content = message.content
            memory_list = []
            try:
                if "```" in content:
                    content = content.split("```")[1].strip()

                if content.startswith("json"):
                    content = content.strip("json")

                for exp_dict in json.loads(content):
                    when_to_use = exp_dict.get("when_to_use", "").strip()
                    memory = exp_dict.get("memory", "").strip()
                    if when_to_use and memory:
                        memory_list.append(
                            StateMemory(
                                workspace_id=self.context.get("workspace_id", ""),
                                when_to_use=when_to_use,
                                content=memory,
                                target=state_name,
                                author=getattr(self.llm, "model_name", "system"),
                            ),
                        )

                return memory_list

            except Exception as e:
                logger.exception(f"parse state summary content failed!\n{content}")
                raise e

        return await self.llm.achat(
            messages=[FlowMessage(role=Role.USER, content=summary_prompt)],
            callback_fn=parse_content,
        )
