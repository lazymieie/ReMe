"""Validation operation for state memory quality control."""

import json
import re
from typing import Any, Dict, List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message as FlowMessage
from loguru import logger

from reme_ai.schema.memory import BaseMemory


@C.register_op()
class StateMemoryValidationOp(BaseAsyncOp):
    """Validate quality of extracted state memories."""

    file_path: str = __file__

    async def async_execute(self):
        state_memories: List[BaseMemory] = []
        state_memories.extend(self.context.get("success_state_memories", []))
        state_memories.extend(self.context.get("failure_state_memories", []))
        state_memories.extend(self.context.get("comparative_state_memories", []))

        if not state_memories:
            logger.info("No state memories found for validation")
            self.context.response.answer = json.dumps([], ensure_ascii=False)
            self.context.response.metadata["memory_list"] = []
            return

        validated_state_memories = []
        for state_memory in state_memories:
            validation_result = await self._validate_single_state_memory(state_memory)
            if validation_result and validation_result.get("is_valid", False):
                state_memory.score = validation_result.get("score", 0.0)
                validated_state_memories.append(state_memory)
            else:
                reason = validation_result.get("reason", "Unknown reason") if validation_result else "Validation failed"
                logger.warning(f"State memory validation failed: {reason}")

        logger.info(f"Validated {len(validated_state_memories)} out of {len(state_memories)} state memories")
        self.context.response.answer = json.dumps(
            [x.model_dump() for x in validated_state_memories],
            ensure_ascii=False,
        )
        self.context.response.metadata["memory_list"] = validated_state_memories

    async def _validate_single_state_memory(self, state_memory: BaseMemory) -> Dict[str, Any]:
        try:
            prompt = self.prompt_format(
                prompt_name="state_memory_validation_prompt",
                condition=state_memory.when_to_use,
                state_memory_content=state_memory.content,
                state_name=getattr(state_memory, "target", ""),
            )

            def parse_validation(message: FlowMessage) -> Dict[str, Any]:
                return self._parse_validation_response(
                    message.content,
                    self.op_params.get("validation_threshold", 0.5),
                )

            return await self.llm.achat(
                messages=[FlowMessage(role=Role.USER, content=prompt)],
                callback_fn=parse_validation,
            )
        except Exception as e:
            logger.error(f"LLM state validation failed: {e}")
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": "",
                "reason": f"LLM validation error: {str(e)}",
            }

    @staticmethod
    def _parse_validation_response(response_content: str, validation_threshold: float) -> Dict[str, Any]:
        try:
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            json_blocks = re.findall(json_pattern, response_content)
            if json_blocks:
                parsed = json.loads(json_blocks[0])
            else:
                parsed = json.loads(response_content)

            is_valid = parsed.get("is_valid", True)
            score = parsed.get("score", 0.5)

            return {
                "is_valid": is_valid and score >= validation_threshold,
                "score": score,
                "feedback": response_content,
                "reason": (
                    ""
                    if (is_valid and score >= validation_threshold)
                    else f"Low validation score ({score:.2f}) or marked as invalid"
                ),
            }
        except Exception as e_inner:
            logger.exception(f"Error parsing state validation response: {e_inner}")
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": "",
                "reason": f"Parse error: {str(e_inner)}",
            }
