"""Unit tests for state summary ops helper logic."""

from types import SimpleNamespace

from reme_ai.schema import Message, Trajectory
from reme_ai.schema.memory import StateMemory
from reme_ai.summary.state.comparative_extraction_op import StateComparativeExtractionOp
from reme_ai.summary.state.failure_extraction_op import StateFailureExtractionOp
from reme_ai.summary.state.memory_deduplication_op import StateMemoryDeduplicationOp
from reme_ai.summary.state.memory_validation_op import StateMemoryValidationOp
from reme_ai.summary.state.success_extraction_op import StateSuccessExtractionOp
from reme_ai.summary.state.summary_state_memory_op import SummaryStateMemoryOp
from reme_ai.summary.state.trajectory_preprocess_op import StateTrajectoryPreprocessOp
from reme_ai.summary.state.trajectory_segmentation_op import StateTrajectorySegmentationOp


def test_state_preprocess_classifies_success_and_failure():
    op = object.__new__(StateTrajectoryPreprocessOp)
    op.op_params = {"success_threshold": 0.8}

    trajectories = [
        Trajectory(messages=[Message(role="assistant", content="ok")], score=1.0, metadata={}),
        Trajectory(messages=[Message(role="assistant", content="bad")], score=0.2, metadata={}),
    ]

    classified = op._classify_trajectories(trajectories)

    assert len(classified["success"]) == 1
    assert len(classified["failure"]) == 1
    assert len(classified["all"]) == 2


def test_state_segmentation_parses_fenced_json():
    response = """```json\n{\"segment_points\": [2, 5, 7]}\n```"""
    assert StateTrajectorySegmentationOp._parse_segmentation_response(response) == [2, 5, 7]


def test_state_segmentation_parses_raw_json():
    response = "{\"segment_points\": [3, 6]}"
    assert StateTrajectorySegmentationOp._parse_segmentation_response(response) == [3, 6]


def test_state_segmentation_parses_embedded_json_without_reasoning_number_pollution():
    response = (
        "Reasoning: steps 1 and 2 are setup, step 3 is blocker, steps 4 and 5 recover.\n"
        "{\"reasoning\": \"group setup then recovery\", \"segment_points\": [2, 5]}"
    )
    assert StateTrajectorySegmentationOp._parse_segmentation_response(response) == [2, 5]


def test_success_extraction_builds_state_memories_from_prompt_json():
    op = object.__new__(StateSuccessExtractionOp)
    op.context = {"workspace_id": "ws"}
    op.llm = SimpleNamespace(model_name="tester")

    content = """
```json
[
  {
    "reasoning": "state recognized",
    "when_to_use": "The agent is trying to log in and observed a captcha prompt.",
    "experience": "Pause submission and complete verification before retrying."
  }
]
```
"""

    memories = op._build_state_memories(content, "login_page_state")

    assert len(memories) == 1
    assert memories[0].target == "login_page_state"
    assert memories[0].when_to_use.startswith("The agent is trying to log in")


def test_failure_extraction_builds_state_memories_from_prompt_json():
    op = object.__new__(StateFailureExtractionOp)
    op.context = {"workspace_id": "ws"}
    op.llm = SimpleNamespace(model_name="tester")

    content = """
```json
[
  {
    "reasoning": "loop trap",
    "when_to_use": "The agent is trying to submit and observed repeated unchanged error state.",
    "experience": "DO NOT retry blindly. INSTEAD, inspect the blocking signal and switch recovery strategy."
  }
]
```
"""

    memories = op._build_state_memories(content, "submit_state")

    assert len(memories) == 1
    assert memories[0].content.startswith("DO NOT retry blindly")


def test_comparative_extraction_selects_highest_and_lowest_scores():
    trajectories = [
        Trajectory(messages=[Message(role="assistant", content="mid")], score=0.5, metadata={}),
        Trajectory(messages=[Message(role="assistant", content="high")], score=0.9, metadata={}),
        Trajectory(messages=[Message(role="assistant", content="low")], score=0.1, metadata={}),
    ]

    highest, lowest = StateComparativeExtractionOp._find_highest_lowest_scoring_trajectories(trajectories)

    assert highest.score == 0.9
    assert lowest.score == 0.1


def test_comparative_extraction_similarity_search_returns_pairs():
    op = object.__new__(StateComparativeExtractionOp)
    op.op_params = {
        "enable_similarity_comparison": True,
        "max_similarity_sequences": 5,
        "similarity_threshold": 0.5,
        "max_similarity_pairs": 3,
    }
    op.context = {"state_name": "default_state", "workspace_id": "ws"}
    op.vector_store = SimpleNamespace(
        embedding_model=SimpleNamespace(
            get_embeddings=lambda texts: [[1.0, 0.0] if "captcha" in text else [0.0, 1.0] for text in texts],
        ),
    )

    success = [
        Trajectory(
            messages=[Message(role="assistant", content="captcha appears and verification succeeds")],
            score=1.0,
            metadata={"state_name": "login_state"},
        ),
    ]
    failure = [
        Trajectory(
            messages=[Message(role="assistant", content="captcha appears and retries loop")],
            score=0.0,
            metadata={"state_name": "error_state"},
        ),
    ]

    pairs = op._find_similar_step_sequences(success, failure)

    assert len(pairs) == 1
    assert pairs[0][3] == "login_state"


def test_state_validation_parser_accepts_valid_json():
    response = """
```json
{
  "is_valid": true,
  "score": 0.85,
  "feedback": "well formatted",
  "recommendations": ""
}
```
"""
    parsed = StateMemoryValidationOp._parse_validation_response(response, validation_threshold=0.5)

    assert parsed["is_valid"] is True
    assert parsed["score"] == 0.85


def test_state_validation_parser_rejects_low_score():
    response = '{"is_valid": true, "score": 0.2, "feedback": "too weak", "recommendations": "improve"}'
    parsed = StateMemoryValidationOp._parse_validation_response(response, validation_threshold=0.5)

    assert parsed["is_valid"] is False
    assert "Low validation score" in parsed["reason"]


def test_state_dedup_cosine_similarity_handles_identical_vectors():
    similarity = StateMemoryDeduplicationOp._calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert similarity == 1.0


def test_summary_state_memory_fallback_preserves_plain_format():
    op = object.__new__(SummaryStateMemoryOp)
    op.context = {"workspace_id": "ws", "description": "login"}
    op.llm = SimpleNamespace(model_name="tester")

    content = """
```json
[
  {
    "when_to_use": "The agent is trying to log in and observed a captcha prompt.",
    "memory": "Pause and complete verification before continuing."
  }
]
```
"""

    memories = []
    for item in [
        {
            "when_to_use": "The agent is trying to log in and observed a captcha prompt.",
            "memory": "Pause and complete verification before continuing.",
        },
    ]:
        memories.append(
            StateMemory(
                workspace_id="ws",
                when_to_use=item["when_to_use"],
                content=item["memory"],
                target="login_state",
                author="tester",
            ),
        )

    assert memories[0].target == "login_state"
    assert "captcha prompt" in memories[0].when_to_use
