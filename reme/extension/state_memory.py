"""HTTP flow ops for state memory summarization and retrieval."""

from typing import Any

from ..core.enumeration import MemoryType
from ..core.op import BaseOp
from ..core.schema import Message, MemoryNode
from ..memory.vector_based import BaseMemoryAgent, ReMeRetriever, ReMeSummarizer, StateRetriever, StateSummarizer
from ..memory.vector_tools import AddDraftAndRetrieveSimilarMemory, AddHistory, AddMemory, DelegateTask, ReadHistory
from ..memory.vector_tools.record.retrieve_memory import RetrieveMemory


def _ensure_state_target(
    mapping: dict[str, MemoryType],
    state_name: str | list[str],
) -> list[str]:
    """Register state target(s) into memory_target_type_mapping."""
    if isinstance(state_name, str):
        state_names = [state_name]
    elif isinstance(state_name, list):
        state_names = state_name
    else:
        raise RuntimeError("state_name must be str or list[str]")

    for name in state_names:
        if name in mapping:
            assert mapping[name] is MemoryType.STATE
        else:
            mapping[name] = MemoryType.STATE

    return state_names


class SummaryStateMemory(BaseOp):
    """Summarize state memories from message history for HTTP service usage."""

    async def execute(self):
        messages: list[Message | dict] = self.context.get("messages", [])
        state_name: str | list[str] = self.context.get("state_name", "")
        description: str = self.context.get("description", "")
        enable_thinking_params: bool = self.context.get("enable_thinking_params", True)
        retrieve_top_k: int = self.context.get("retrieve_top_k", 20)
        llm_config_name: str = self.context.get("llm_config_name", "default")

        if not messages:
            raise ValueError("messages must not be empty")
        if not state_name:
            raise ValueError("state_name must not be empty")

        format_messages: list[Message] = []
        for message in messages:
            format_messages.append(Message(**message) if isinstance(message, dict) else message)

        memory_targets = _ensure_state_target(self.service_context.memory_target_type_mapping, state_name)

        state_summarizer: BaseMemoryAgent = StateSummarizer(
            llm=llm_config_name,
            tools=[
                AddDraftAndRetrieveSimilarMemory(
                    enable_thinking_params=enable_thinking_params,
                    enable_memory_target=False,
                    enable_when_to_use=False,
                    enable_multiple=True,
                    top_k=retrieve_top_k,
                ),
                AddMemory(
                    enable_thinking_params=enable_thinking_params,
                    enable_memory_target=False,
                    enable_when_to_use=False,
                    enable_multiple=True,
                ),
            ],
        )

        reme_summarizer: BaseMemoryAgent = ReMeSummarizer(
            tools=[AddHistory(), DelegateTask(memory_agents=[state_summarizer])],
        )

        result = await reme_summarizer.call(
            messages=format_messages,
            description=description,
            service_context=self.service_context,
            memory_targets=memory_targets,
        )

        memory_list: list[MemoryNode] = result.get("answer", [])
        answer = "\n".join(
            memory.format(include_memory_id=False, include_when_to_use=True, include_content=True)
            for memory in memory_list
        )

        return {
            "answer": answer,
            "success": result.get("success", True),
            "memory_list": memory_list,
            "messages": result.get("messages", []),
            "tools": result.get("tools", []),
        }


class RetrieveStateMemory(BaseOp):
    """Retrieve state memories for HTTP service usage."""

    async def execute(self):
        query: str = self.context.get("query", "")
        messages: list[dict] | None = self.context.get("messages", None)
        state_name: str | list[str] = self.context.get("state_name", "")
        description: str = self.context.get("description", "")
        enable_thinking_params: bool = self.context.get("enable_thinking_params", True)
        retrieve_top_k: int = self.context.get("retrieve_top_k", 20)
        llm_config_name: str = self.context.get("llm_config_name", "default")

        if not query and not messages:
            raise ValueError("Either query or messages must be provided")
        if not state_name:
            raise ValueError("state_name must not be empty")

        memory_targets = _ensure_state_target(self.service_context.memory_target_type_mapping, state_name)

        state_retriever: BaseMemoryAgent = StateRetriever(
            llm=llm_config_name,
            tools=[
                RetrieveMemory(
                    top_k=retrieve_top_k,
                    enable_thinking_params=enable_thinking_params,
                    enable_time_filter=False,
                    enable_multiple=True,
                ),
                ReadHistory(
                    enable_thinking_params=enable_thinking_params,
                    enable_multiple=True,
                ),
            ],
        )

        reme_retriever: BaseMemoryAgent = ReMeRetriever(
            tools=[DelegateTask(memory_agents=[state_retriever])],
        )

        result = await reme_retriever.call(
            query=query,
            messages=messages,
            description=description,
            service_context=self.service_context,
            memory_targets=memory_targets,
        )

        retrieved_nodes: list[MemoryNode] = result.get("retrieved_nodes", [])
        return {
            "answer": result.get("answer", ""),
            "success": result.get("success", True),
            "memory_list": retrieved_nodes,
            "messages": result.get("messages", []),
            "tools": result.get("tools", []),
        }
