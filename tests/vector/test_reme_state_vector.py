"""测试 ReMe 的 state memory 功能"""

import asyncio

from reme import ReMe


async def main():
    """测试 state memory 的总结、检索和管理流程"""
    reme = ReMe(
        working_dir=".reme_state_demo",
        default_llm_config={
            "backend": "openai",
            "model_name": "qwen3.5-plus",
        },
        default_embedding_model_config={
            "backend": "openai",
            "model_name": "text-embedding-v4",
            "dimensions": 1024,
        },
        default_vector_store_config={
            "backend": "local",
        },
    )
    await reme.start()

    state_name = "login_page_state"

    messages = [
        {"role": "user", "content": "请登录后台系统", "time_created": "2026-03-22 10:00:00"},
        {"role": "assistant", "content": "已打开登录页", "time_created": "2026-03-22 10:00:03"},
        {"role": "assistant", "content": "已输入用户名和密码", "time_created": "2026-03-22 10:00:08"},
        {"role": "assistant", "content": "点击登录按钮", "time_created": "2026-03-22 10:00:12"},
        {"role": "assistant", "content": "页面未跳转，仍停留在登录页", "time_created": "2026-03-22 10:00:15"},
        {"role": "assistant", "content": "检测到验证码弹窗", "time_created": "2026-03-22 10:00:17"},
        {"role": "assistant", "content": "再次点击登录仍无变化", "time_created": "2026-03-22 10:00:21"},
        {"role": "assistant", "content": "刷新页面并等待验证码完成", "time_created": "2026-03-22 10:00:28"},
        {"role": "assistant", "content": "重新提交后已进入首页", "time_created": "2026-03-22 10:00:36"},
    ]

    print("\n=== 1. 自动总结 state memory ===")
    summary_result = await reme.summarize_memory(
        messages=messages,
        state_name=state_name,
        return_dict=True,
    )
    print(summary_result.get("answer", ""))

    print("\n=== 2. 检索 state memory ===")
    retrieved = await reme.retrieve_memory(
        query="登录后页面还停在原地，并且出现了验证码，这种状态下应该怎么处理？",
        state_name=state_name,
    )
    print(retrieved)

    print("\n=== 3. 手动添加一条 state memory ===")
    memory_node = await reme.add_memory(
        memory_content=(
            "当登录页出现验证码弹窗且重复点击登录没有状态变化时，说明流程已进入风控校验或无效重试状态；"
            "此时应停止重复提交，优先检查 DOM、URL 或等待验证完成。"
        ),
        when_to_use="当页面停留在登录页，出现验证码，并且重复提交无效时",
        state_name=state_name,
    )
    print(memory_node)

    print("\n=== 4. 再次检索，观察手动经验是否被召回 ===")
    retrieved_again = await reme.retrieve_memory(
        query="页面停留在登录页并出现验证码，重复提交也没有反应",
        state_name=state_name,
    )
    print(retrieved_again)

    print("\n=== 5. 列出当前 state memory ===")
    memory_list = await reme.list_memory(
        state_name=state_name,
        limit=20,
        sort_key="time_created",
        reverse=True,
    )
    for i, node in enumerate(memory_list, start=1):
        print(f"\n--- memory {i} ---")
        print(node)

    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
