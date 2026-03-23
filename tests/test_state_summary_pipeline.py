"""Regression tests for the state-summary pipeline helpers."""

from reme_ai.schema import Message
from reme_ai.schema.memory import StateMemory
from reme_ai.summary.state.memory_deduplication_op import StateMemoryDeduplicationOp
from reme_ai.summary.state.trajectory_preprocess_op import (
    DEFAULT_STATE_NAME,
    StateTrajectoryPreprocessOp,
)


def test_state_preprocess_builds_single_trajectory_from_messages():
    """Raw messages should be normalized into one synthetic trajectory."""
    messages = [
        Message(role="user", content="打开登录页"),
        Message(role="assistant", content="检测到验证码弹窗"),
    ]

    trajectories = StateTrajectoryPreprocessOp._normalize_trajectories(
        messages=messages,
        trajectories=[],
        state_name="login_page_state",
        description="登录流程状态总结",
    )

    assert len(trajectories) == 1
    assert trajectories[0].score == 1.0
    assert trajectories[0].messages == messages
    assert trajectories[0].metadata == {
        "state_name": "login_page_state",
        "query": "登录流程状态总结",
    }


def test_state_preprocess_uses_default_state_name_for_synthetic_trajectory():
    """Empty state names should fall back to the shared default target."""
    messages = [Message(role="assistant", content="仍停留在登录页")]

    trajectories = StateTrajectoryPreprocessOp._normalize_trajectories(
        messages=messages,
        trajectories=[],
        state_name=DEFAULT_STATE_NAME,
        description="",
    )

    assert trajectories[0].metadata == {"state_name": DEFAULT_STATE_NAME}


def test_state_dedup_cosine_similarity_handles_identical_vectors():
    """State dedup helper should treat identical embeddings as fully similar."""
    similarity = StateMemoryDeduplicationOp._calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert similarity == 1.0


def test_state_memory_instance_keeps_target_for_full_pool_dedup():
    """State memories still keep their target even when dedup checks the full pool."""
    memory = StateMemory(target="login_page_state", when_to_use="出现验证码", content="停止重试")
    assert memory.target == "login_page_state"
