"""Tool-use trajectory recording for fx-1 (K3 teacher sessions).

K3 was trained in preserved-thinking-history mode, so trajectories keep
``reasoning_content`` and ``tool_calls`` intact. Admission is fail-closed:
a trajectory enters the corpus only when every harness artifact it produced
passed verification (``verify_ok=True``). Failed-gate trajectories are kept
as negative examples.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class TraceStep(BaseModel):
    """One turn: assistant reasoning + optional harness tool call + result."""

    reasoning_content: str = ""
    assistant_content: str = ""
    tool_call: ToolCall | None = None
    tool_result: str | None = None


class Trajectory(BaseModel):
    """A full research session, admissible to the corpus only if verified."""

    session_id: str
    user_intent: str
    steps: list[TraceStep]
    verify_ok: bool = False
    artifact_receipts: list[str] = Field(default_factory=list)
    recorded_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    @property
    def sha256(self) -> str:
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_sft_messages(self, system: str) -> list[dict[str, str]]:
        """Flatten to chat messages, preserving reasoning and tool calls."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.user_intent},
        ]
        for step in self.steps:
            content = step.assistant_content
            if step.reasoning_content:
                content = (
                    f"<reasoning>{step.reasoning_content}</reasoning>\n{content}"
                )
            if step.tool_call is not None:
                content += (
                    f"\n<tool_call>{step.tool_call.model_dump_json()}</tool_call>"
                )
            messages.append({"role": "assistant", "content": content.strip()})
            if step.tool_result is not None:
                messages.append(
                    {"role": "tool", "content": step.tool_result}
                )
        return messages


class TraceRecorder:
    """Append-only trajectory recorder with verification-gated admission."""

    def __init__(self, out_jsonl: str | Path) -> None:
        self._out = Path(out_jsonl)
        self._out.parent.mkdir(parents=True, exist_ok=True)

    def admit(self, trajectory: Trajectory, system: str) -> bool:
        """Write the trajectory if admissible. Returns admission decision."""
        record = {
            "messages": trajectory.to_sft_messages(system),
            "receipt_sha256": trajectory.sha256,
            "source_path": f"trace:{trajectory.session_id}",
            "negative": not trajectory.verify_ok,
            "artifact_receipts": trajectory.artifact_receipts,
        }
        with self._out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return True
