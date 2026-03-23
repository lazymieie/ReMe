"""Trajectory segmentation operation for state memory generation."""

import json
import re
from typing import List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message as FlowMessage
from loguru import logger

from reme_ai.schema import Message, Trajectory


@C.register_op()
class StateTrajectorySegmentationOp(BaseAsyncOp):
    """Segment trajectories into coherent state episodes."""

    file_path: str = __file__

    async def async_execute(self):
        all_trajectories: List[Trajectory] = self.context.get("all_trajectories", [])
        success_trajectories: List[Trajectory] = self.context.get("success_trajectories", [])
        failure_trajectories: List[Trajectory] = self.context.get("failure_trajectories", [])

        if not all_trajectories:
            logger.warning("No trajectories found in context")
            return

        target_trajectories = self._get_target_trajectories(
            all_trajectories,
            success_trajectories,
            failure_trajectories,
        )

        segmented_count = 0
        for trajectory in target_trajectories:
            trajectory.metadata["segments"] = await self._llm_segment_trajectory(trajectory)
            segmented_count += 1

        logger.info(f"Segmented {segmented_count} state trajectories")

    def _get_target_trajectories(
        self,
        all_trajectories: List[Trajectory],
        success_trajectories: List[Trajectory],
        failure_trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        segment_target = self.op_params.get("segment_target", "all")

        if segment_target == "success":
            return success_trajectories
        if segment_target == "failure":
            return failure_trajectories
        return all_trajectories

    async def _llm_segment_trajectory(self, trajectory: Trajectory) -> List[List[Message]]:
        prompt = self.prompt_format(
            prompt_name="state_step_segmentation_prompt",
            query=trajectory.metadata.get("query", ""),
            state_name=trajectory.metadata.get("state_name", ""),
            trajectory_content=self._format_trajectory_content(trajectory),
            total_steps=len(trajectory.messages),
        )

        def parse_segmentation(message: Message) -> List[List[Message]]:
            segment_points = self._parse_segmentation_response(message.content)
            segments = []
            start_idx = 0

            for end_idx in segment_points:
                if start_idx < end_idx <= len(trajectory.messages):
                    segments.append(trajectory.messages[start_idx:end_idx])
                    start_idx = end_idx

            if start_idx < len(trajectory.messages):
                segments.append(trajectory.messages[start_idx:])

            return segments if segments else [trajectory.messages]

        return await self.llm.achat(
            messages=[FlowMessage(role=Role.USER, content=prompt)],
            callback_fn=parse_segmentation,
            default_value=[trajectory.messages],
        )

    @staticmethod
    def _format_trajectory_content(trajectory: Trajectory) -> str:
        content = ""
        for i, step in enumerate(trajectory.messages):
            content += f"Step {i + 1} ({step.role.value}):\n{step.content}\n\n"
        return content

    @staticmethod
    def _parse_segmentation_response(response: str) -> List[int]:
        segment_points = StateTrajectorySegmentationOp._extract_segment_points_from_json(response)

        if not segment_points:
            numbers = re.findall(r"\b\d+\b", response)
            segment_points = [int(num) for num in numbers if int(num) > 0]

        return sorted(list(set(segment_points)))

    @staticmethod
    def _extract_segment_points_from_json(response: str) -> List[int]:
        """Extract segment points from fenced JSON, raw JSON, or embedded JSON."""
        candidate_payloads = []

        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        candidate_payloads.extend(re.findall(json_pattern, response))
        candidate_payloads.append(response.strip())

        object_pattern = r"(\{[\s\S]*?\"segment_points\"[\s\S]*?\})"
        candidate_payloads.extend(re.findall(object_pattern, response))

        array_pattern = r"(\[\s*\d+(?:\s*,\s*\d+)*\s*\])"
        candidate_payloads.extend(re.findall(array_pattern, response))

        for payload in candidate_payloads:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict) and "segment_points" in parsed:
                return StateTrajectorySegmentationOp._normalize_segment_points(parsed["segment_points"])
            if isinstance(parsed, list):
                return StateTrajectorySegmentationOp._normalize_segment_points(parsed)

        return []

    @staticmethod
    def _normalize_segment_points(segment_points: List[int] | List[str]) -> List[int]:
        normalized_points = []
        for point in segment_points:
            try:
                point_int = int(point)
            except (TypeError, ValueError):
                continue
            if point_int > 0:
                normalized_points.append(point_int)
        return normalized_points
