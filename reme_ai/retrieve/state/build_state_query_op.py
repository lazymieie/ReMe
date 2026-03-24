"""Build retrieval queries for state memory search."""

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from flowllm.core.utils import merge_messages_content
from loguru import logger


@C.register_op()
class BuildStateQueryOp(BaseAsyncOp):
    """Build a state-aware retrieval query from explicit query or recent messages."""

    file_path: str = __file__

    async def async_execute(self):
        if "query" in self.context and self.context.query:
            query = self.context.query

        elif "messages" in self.context:
            if self.op_params.get("enable_llm_build", True):
                execution_process = merge_messages_content(self.context.messages)
                prompt = self.prompt_format(prompt_name="query_build", execution_process=execution_process)
                message = await self.llm.achat(messages=[Message(role=Role.USER, content=prompt)])
                query = message.content
            else:
                context_parts = []
                message_summaries = []
                for message in self.context.messages[-3:]:
                    content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    message_summaries.append(f"- {message.role.value}: {content}")
                if message_summaries:
                    context_parts.append("Recent messages:\n" + "\n".join(message_summaries))
                query = "\n\n".join(context_parts)

        else:
            raise RuntimeError("query or messages is required!")

        logger.info(f"build.state_query={query}")
        self.context.query = query
