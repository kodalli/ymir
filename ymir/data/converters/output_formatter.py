"""Export trajectories to training-ready formats."""

import json
from pathlib import Path
from typing import Iterator

from ymir.core import MessageRole, Trajectory


class TrainingDataExporter:
    """Export trajectories to various training formats."""

    def export_jsonl(
        self,
        trajectories: Iterator[Trajectory],
        output_path: str,
        format: str = "chatml",
    ) -> int:
        """
        Export trajectories to JSONL format.

        Formats:
        - "chatml": ChatML format for training (messages array)
        - "standard": Full trajectory with all metadata
        - "simple": Minimal format with just messages
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for traj in trajectories:
                if format == "chatml":
                    record = self._to_chatml(traj)
                elif format == "simple":
                    record = self._to_simple(traj)
                else:
                    record = self._to_standard(traj)

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count

    def _to_chatml(self, traj: Trajectory) -> dict:
        """
        Convert to ChatML format.

        Format compatible with common training frameworks.
        """
        messages = [{"role": "system", "content": traj.system_prompt}]

        for msg in traj.messages:
            if msg.role == MessageRole.TOOL_RESULT:
                messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "name": msg.tool_result.tool_name if msg.tool_result else "unknown",
                    }
                )
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        return {"messages": messages, "tools": traj.tools}

    def _to_standard(self, traj: Trajectory) -> dict:
        """Full format with all metadata."""
        return {
            "id": traj.id,
            "scenario_id": traj.scenario_id,
            "scenario_description": traj.scenario_description,
            "tools": traj.tools,
            "system_prompt": traj.system_prompt,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
                    if msg.tool_calls
                    else None,
                    "tool_result": msg.tool_result.model_dump() if msg.tool_result else None,
                }
                for msg in traj.messages
            ],
            "metadata": {
                "source": traj.source,
                "original_source": traj.original_source,
                "status": traj.status.value,
                "quality_score": traj.quality_score,
                "created_at": traj.created_at.isoformat(),
            },
        }

    def _to_simple(self, traj: Trajectory) -> dict:
        """Minimal format with just the conversation."""
        return {
            "system": traj.system_prompt,
            "messages": [
                {"role": msg.role.value, "content": msg.content}
                for msg in traj.messages
            ],
        }

    def export_for_training(
        self,
        trajectories: Iterator[Trajectory],
        output_path: str,
    ) -> int:
        """
        Export in the exact format used by llm-agent-training.

        Format: {input, output} where input is system + context and output is the response.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for traj in trajectories:
                # Build input from system + messages (except last assistant message)
                context_parts = [traj.system_prompt]

                for i, msg in enumerate(traj.messages):
                    # Skip the last message if it's an assistant response (that's our target)
                    is_last_assistant = (
                        i == len(traj.messages) - 1 and msg.role == MessageRole.ASSISTANT
                    )
                    if is_last_assistant:
                        output = msg.content
                        break

                    if msg.role == MessageRole.USER:
                        context_parts.append(f"User: {msg.content}")
                    elif msg.role == MessageRole.ASSISTANT:
                        context_parts.append(f"Assistant: {msg.content}")
                    elif msg.role == MessageRole.TOOL_RESULT:
                        tool_name = msg.tool_result.tool_name if msg.tool_result else "unknown"
                        context_parts.append(f"Tool result [{tool_name}]: {msg.content}")
                else:
                    # No final assistant message found
                    continue

                input_text = "\n\n".join(context_parts) + "\n\nResponse:"

                record = {"input": input_text, "output": output}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count
