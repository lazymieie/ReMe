"""Regression tests for state-memory CRUD entrypoints on ``ReMe``."""

import pytest

from reme import ReMe
from reme.core.enumeration import MemoryType


class _FakeMemoryHandler:
    """Minimal async handler used to capture ReMe CRUD calls."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def add(self, **kwargs):
        self.calls.append(("add", kwargs))
        return {"op": "add", **kwargs}

    async def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"op": "update", **kwargs}

    async def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return [{"op": "list", **kwargs}]


def test_resolve_memory_target_supports_state_name():
    """State memories should resolve to the dedicated state type and target."""
    memory_type, memory_target = ReMe._resolve_memory_target(state_name="login_page_state")

    assert memory_type is MemoryType.STATE
    assert memory_target == "login_page_state"


@pytest.mark.asyncio
async def test_add_memory_routes_state_name_to_state_handler(tmp_path, monkeypatch):
    """Manual state memory insertion should not require user/task/tool names."""
    reme = ReMe(working_dir=str(tmp_path), enable_profile=False)
    handler = _FakeMemoryHandler()

    requested_targets: list[str] = []

    def _fake_get_memory_handler(memory_target: str):
        requested_targets.append(memory_target)
        return handler

    monkeypatch.setattr(reme, "get_memory_handler", _fake_get_memory_handler)

    result = await reme.add_memory(
        memory_content="验证码出现后不要重复提交。",
        when_to_use="登录页停留原地且出现验证码时",
        state_name="login_page_state",
    )

    assert result["op"] == "add"
    assert requested_targets == ["login_page_state"]
    assert handler.calls == [
        (
            "add",
            {
                "content": "验证码出现后不要重复提交。",
                "when_to_use": "登录页停留原地且出现验证码时",
                "message_time": "",
                "ref_memory_id": "",
                "author": "",
                "score": 0.0,
            },
        ),
    ]
    assert reme.service_context.memory_target_type_mapping["login_page_state"] is MemoryType.STATE


@pytest.mark.asyncio
async def test_update_and_list_memory_accept_state_name(tmp_path, monkeypatch):
    """State-name CRUD helpers should resolve and pass through handler arguments."""
    reme = ReMe(working_dir=str(tmp_path), enable_profile=False)
    handler = _FakeMemoryHandler()

    requested_targets: list[str] = []

    def _fake_get_memory_handler(memory_target: str):
        requested_targets.append(memory_target)
        return handler

    monkeypatch.setattr(reme, "get_memory_handler", _fake_get_memory_handler)

    updated = await reme.update_memory(
        memory_id="memory-1",
        state_name="login_page_state",
        memory_content="等待验证码完成后再提交。",
        score=0.8,
    )
    listed = await reme.list_memory(
        state_name="login_page_state",
        filters={"author": "tester"},
        limit=5,
        sort_key="time_created",
        reverse=False,
    )

    assert updated["op"] == "update"
    assert listed == [
        {
            "op": "list",
            "filters": {"author": "tester"},
            "limit": 5,
            "sort_key": "time_created",
            "reverse": False,
        },
    ]
    assert requested_targets == ["login_page_state", "login_page_state"]
    assert handler.calls == [
        (
            "update",
            {
                "memory_id": "memory-1",
                "content": "等待验证码完成后再提交。",
                "when_to_use": None,
                "message_time": None,
                "ref_memory_id": None,
                "author": None,
                "score": 0.8,
            },
        ),
        (
            "list",
            {
                "filters": {"author": "tester"},
                "limit": 5,
                "sort_key": "time_created",
                "reverse": False,
            },
        ),
    ]
    assert reme.service_context.memory_target_type_mapping["login_page_state"] is MemoryType.STATE
