"""State memory summarizer agent for extracting and storing state-aware experiences."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message


class StateSummarizer(BaseMemoryAgent):
    """Extract and store state memories from execution trajectories.

    State memories capture "what state was observed and what it implies", including:
    - Environment states and preconditions before an action
    - Intermediate progress checkpoints and status transitions
    - Signals that indicate whether to continue, retry, or switch strategy
    - State-action implications that are reusable in similar situations
    """

    memory_type: MemoryType = MemoryType.STATE

    async def build_messages(self) -> list[Message]:
        """Build messages for state memory extraction."""
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message",
                    context=self.context.history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
        ]

    async def _acting_step(
        self,
        assistant_message: Message,
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        """Execute tool calls with memory context."""
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            stage=stage,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            history_node=self.history_node,
            author=self.author,
            retrieved_nodes=self.retrieved_nodes,
            **kwargs,
        )

    async def execute(self):
        """Execute state memory extraction."""
        for i, tool in enumerate(self.tools):
            logger.info(f"[{self.__class__.__name__}] tool_call[{i}]={tool.tool_call.simple_input_dump(as_dict=False)}")

        messages = await self.build_messages()
        for i, message in enumerate(messages):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__}] role={role} {message.simple_dump(as_dict=False)}")

        tools, messages, success = await self.react(messages, self.tools)

        answer = messages[-1].content if success and messages else ""
        memory_nodes = []
        for tool in tools:
            if tool.memory_nodes:
                memory_nodes.extend(tool.memory_nodes)

        return {
            "answer": answer,
            "success": success,
            "messages": messages,
            "tools": tools,
            "memory_nodes": memory_nodes,
        }
