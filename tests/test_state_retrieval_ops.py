"""Unit tests for state retrieval workflow ops in ``reme_ai``."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from flowllm.core.schema import Message, VectorNode

from reme_ai.retrieve.state.build_state_query_op import BuildStateQueryOp
from reme_ai.retrieve.state.retrieve_state_memory_op import RetrieveStateMemoryOp
from reme_ai.retrieve.state.state_rewrite_memory_op import StateRewriteMemoryOp
from reme_ai.retrieve.state.state_rerank_memory_op import StateRerankMemoryOp
from reme_ai.retrieve.task.rerank_memory_op import RerankMemoryOp
from reme_ai.schema.memory import StateMemory


class _AttrDict(dict):
    """Small dict helper that also exposes attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _response():
    return SimpleNamespace(answer="", success=True, metadata={})


def test_retrieve_state_memory_flow_uses_state_workflow_ops():
    """Service config should point retrieve_state_memory at the new state workflow."""
    config_text = Path("reme_ai/config/default.yaml").read_text(encoding="utf-8")

    assert "retrieve_state_memory:" in config_text
    assert "BuildStateQueryOp" in config_text
    assert "RetrieveStateMemoryOp" in config_text
    assert "StateRerankMemoryOp" in config_text
    assert "StateRewriteMemoryOp" in config_text


@pytest.mark.asyncio
async def test_build_state_query_op_uses_recent_messages_when_llm_disabled():
    """State query builder should create a contextual query from recent messages."""
    op = object.__new__(BuildStateQueryOp)
    op.op_params = {"enable_llm_build": False}
    op.context = _AttrDict(
        messages=[
            Message(role="assistant", content="已打开登录页"),
            Message(role="assistant", content="页面仍停留原地"),
            Message(role="assistant", content="检测到验证码弹窗"),
        ],
        response=_response(),
    )

    await op.async_execute()

    assert "Recent messages" in op.context.query
    assert "验证码弹窗" in op.context.query


@pytest.mark.asyncio
async def test_retrieve_state_memory_op_filters_by_state_target_and_deduplicates():
    """State retrieval should only keep matching targets and drop duplicate content."""
    op = object.__new__(RetrieveStateMemoryOp)
    op.context = _AttrDict(
        query="登录页出现验证码并停留原地",
        state_name="login_page_state",
        workspace_id="ws",
        top_k=2,
        response=_response(),
    )

    op.vector_store = SimpleNamespace(
        async_search=_fake_async_search(
            [
                _state_node(
                    target="login_page_state",
                    when_to_use="登录页停留原地且出现验证码",
                    content="停止重复提交，先完成验证。",
                ),
                _state_node(
                    target="other_state",
                    when_to_use="其他状态",
                    content="这条不应该被保留",
                ),
                _state_node(
                    target="login_page_state",
                    when_to_use="登录页停留原地且出现验证码",
                    content="停止重复提交，先完成验证。",
                ),
                _state_node(
                    target="login_page_state",
                    when_to_use="验证码完成后准备再次提交",
                    content="提交前先确认页面状态已变化。",
                ),
            ],
        ),
    )

    await op.async_execute()

    assert op.context.response.success is True
    memory_list = op.context.response.metadata["memory_list"]
    assert len(memory_list) == 2
    assert all(memory.target == "login_page_state" for memory in memory_list)
    assert "State Target: login_page_state" in op.context.response.answer
    assert "其他状态" not in op.context.response.answer


def test_state_rerank_memory_formats_candidates_with_state_content():
    """State rerank op inherits ranking logic but should preserve candidate formatting."""
    memories = [
        StateMemory(target="login_page_state", when_to_use="出现验证码", content="停止重试"),
        StateMemory(target="login_page_state", when_to_use="验证完成", content="检查页面跳转"),
    ]

    formatted = StateRerankMemoryOp._format_candidates_for_rerank(memories)

    assert "Condition: 出现验证码" in formatted
    assert "Experience: 停止重试" in formatted


def test_rerank_response_parses_raw_json_before_falling_back_to_numbers():
    """Raw JSON rerank responses with reasoning should not be polluted by stray numbers."""
    response = (
        '{"reasoning":"Candidate 2 best matches. Candidate 0 is second.",'
        '"ranked_indices":[2,0,1]}'
    )

    parsed = RerankMemoryOp._parse_rerank_response(response)

    assert parsed == [2, 0, 1]


def test_state_rewrite_memory_formats_state_guidance():
    """State rewrite op should expose trigger/action structure even without LLM rewrite."""
    memories = [
        StateMemory(
            target="login_page_state",
            when_to_use="登录页停留原地且出现验证码",
            content="停止重复提交，先等待验证完成，再检查 URL 或 DOM 是否变化。",
        ),
    ]

    formatted = StateRewriteMemoryOp._format_memories_for_context(memories)

    assert "Observed State / Trigger: 登录页停留原地且出现验证码" in formatted
    assert "Implication / Recommended Action: 停止重复提交" in formatted


def test_rewrite_memory_normalizes_structured_rewritten_context():
    """Structured rewritten_context payloads should not break string post-processing."""
    normalized = StateRewriteMemoryOp._normalize_rewritten_context(
        {"summary": "停止重复提交", "next_step": "先处理验证码"},
    )

    assert "停止重复提交" in normalized
    assert "先处理验证码" in normalized


def test_rewrite_memory_prefers_final_text_fields():
    """Structured payloads should extract the final user-facing text instead of raw JSON."""
    normalized = StateRewriteMemoryOp._normalize_rewritten_context(
        {
            "reasoning": "当前状态被验证码阻塞。",
            "content": "停止重复提交，先完成验证码，再检查页面状态。",
        },
    )

    assert normalized == "停止重复提交，先完成验证码，再检查页面状态。"


@pytest.mark.asyncio
async def test_state_rewrite_preserves_no_match_answer():
    """Empty recall results should keep the no-match answer instead of blanking it out."""
    op = object.__new__(StateRewriteMemoryOp)
    op.context = _AttrDict(
        response=SimpleNamespace(
            answer="No matching state memories found",
            success=False,
            metadata={"memory_list": []},
        ),
    )

    await op.async_execute()

    assert op.context.response.answer == "No matching state memories found"


def _state_node(target: str, when_to_use: str, content: str) -> VectorNode:
    memory = StateMemory(
        workspace_id="ws",
        target=target,
        when_to_use=when_to_use,
        content=content,
        author="tester",
    )
    return memory.to_vector_node()


def _fake_async_search(nodes):
    async def _search(**kwargs):
        return nodes

    return _search
