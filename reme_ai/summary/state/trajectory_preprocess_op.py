"""Trajectory preprocessing for state memory generation."""

from typing import Dict, List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from loguru import logger

from reme_ai.schema import Message, Trajectory

DEFAULT_STATE_NAME = "default_state"


@C.register_op()
class StateTrajectoryPreprocessOp(BaseAsyncOp):
    """Normalize input and classify trajectories into success and failure."""

    file_path: str = __file__

    async def async_execute(self):
        trajectories: list = self.context.get("trajectories", [])
        messages: list = self.context.get("messages", [])
        state_name: str = self.context.get("state_name", "") or DEFAULT_STATE_NAME
        description: str = self.context.get("description", "")

        normalized_trajectories = self._normalize_trajectories(trajectories, messages, state_name, description)
        if not normalized_trajectories:
            self.context.response.answer = "trajectories or messages is required"
            self.context.response.success = False
            return

        classified = self._classify_trajectories(normalized_trajectories)
        self.context.state_name = state_name
        self.context.success_trajectories = classified["success"]
        self.context.failure_trajectories = classified["failure"]
        self.context.all_trajectories = classified["all"]

        logger.info(
            f"Classified state trajectories - Success: {len(classified['success'])}, "
            f"Failure: {len(classified['failure'])}, All: {len(classified['all'])}",
        )

    def _classify_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
        success_threshold = self.op_params.get("success_threshold", 1.0)
        success_trajectories: List[Trajectory] = []
        failure_trajectories: List[Trajectory] = []

        for traj in trajectories:
            if traj.score >= success_threshold:
                success_trajectories.append(traj)
            else:
                failure_trajectories.append(traj)

        return {
            "success": success_trajectories,
            "failure": failure_trajectories,
            "all": trajectories,
        }

    @staticmethod
    def _normalize_trajectories(
        trajectories: list,
        messages: list,
        state_name: str,
        description: str,
    ) -> List[Trajectory]:
        if trajectories:
            normalized_trajectories: List[Trajectory] = [Trajectory(**x) if isinstance(x, dict) else x for x in trajectories]
            for trajectory in normalized_trajectories:
                trajectory.metadata.setdefault("state_name", state_name)
                if description:
                    trajectory.metadata.setdefault("query", description)
            return normalized_trajectories

        if not messages:
            return []

        normalized_messages: List[Message] = [Message(**x) if isinstance(x, dict) else x for x in messages]
        return [
            Trajectory(
                messages=normalized_messages,
                score=1.0,
                metadata=StateTrajectoryPreprocessOp._build_metadata(state_name, description),
            ),
        ]

    @staticmethod
    def _build_metadata(state_name: str, description: str) -> Dict[str, str]:
        metadata: Dict[str, str] = {"state_name": state_name}
        if description:
            metadata["query"] = description
        return metadata
