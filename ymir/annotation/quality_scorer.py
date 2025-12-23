"""Automatic quality scoring for trajectories."""

from ymir.core import MessageRole, Trajectory, TOOL_CALL_START
from ymir.core.constants import IDEAL_TURN_RANGE, MAX_TURNS_FOR_QUALITY


class QualityScorer:
    """Automatic quality scoring for trajectories."""

    def score(self, trajectory: Trajectory) -> float:
        """
        Score trajectory quality from 0.0 to 1.0.

        Criteria:
        - Has multiple turns (multi-turn is better for agentic training)
        - Tool calls are properly formatted with <tool_call> tags
        - Has tool results (observations) after tool calls
        - Final response addresses the query (ends with assistant, no pending tool call)
        - No repetitive patterns
        """
        scores = []

        # Turn count score (2-5 turns is ideal, 1 turn is worse, >10 is too many)
        turn_count = trajectory.turn_count()
        scores.append(self._score_turn_count(turn_count))

        # Tool call format score
        scores.append(self._score_tool_format(trajectory))

        # Tool result presence score
        scores.append(self._score_tool_results(trajectory))

        # Completion score (ends with final response, not pending tool call)
        scores.append(self._score_completion(trajectory))

        # Diversity score (no repetition)
        scores.append(self._score_diversity(trajectory))

        return sum(scores) / len(scores)

    def _score_turn_count(self, count: int) -> float:
        """Score based on number of assistant turns."""
        if count < 1:
            return 0.0
        if count == 1:
            return 0.5  # Single turn is okay but not ideal for agentic training
        if IDEAL_TURN_RANGE[0] <= count <= IDEAL_TURN_RANGE[1]:
            return 1.0  # Ideal range
        if count <= MAX_TURNS_FOR_QUALITY:
            return 0.8
        return 0.5  # Too many turns might indicate looping

    def _score_tool_format(self, traj: Trajectory) -> float:
        """Check that tool calls are properly formatted with <tool_call> tags."""
        score = 1.0
        tool_call_messages = [m for m in traj.messages if m.tool_calls]

        if not tool_call_messages:
            # No tool calls - might be okay for some trajectories
            return 0.7

        for msg in tool_call_messages:
            # Check that content contains properly formatted tool calls
            if TOOL_CALL_START not in msg.content:
                score -= 0.3

        return max(0, score)

    def _score_tool_results(self, traj: Trajectory) -> float:
        """Check that tool calls are followed by tool results."""
        tool_call_count = 0
        tool_result_count = 0

        for msg in traj.messages:
            if msg.tool_calls:
                tool_call_count += len(msg.tool_calls)
            if msg.role == MessageRole.TOOL_RESULT:
                tool_result_count += 1

        if tool_call_count == 0:
            return 0.7  # No tool calls, neutral score

        # Each tool call should have a corresponding result
        if tool_result_count >= tool_call_count:
            return 1.0
        if tool_result_count > 0:
            return tool_result_count / tool_call_count
        return 0.3  # Tool calls without any results

    def _score_completion(self, traj: Trajectory) -> float:
        """Check trajectory has a complete final response."""
        if not traj.messages:
            return 0.0

        last_msg = traj.messages[-1]

        # Should end with assistant message
        if last_msg.role != MessageRole.ASSISTANT:
            return 0.3

        # Final message shouldn't have tool calls (should be the final response)
        if last_msg.tool_calls:
            return 0.5  # Ended with tool call, not final response

        # Has content
        if not last_msg.content.strip():
            return 0.3

        return 1.0

    def _score_diversity(self, traj: Trajectory) -> float:
        """Penalize repetitive content."""
        contents = [m.content for m in traj.messages if m.content]

        if not contents:
            return 0.0

        # Check for exact duplicates
        unique = len(set(contents))
        total = len(contents)

        if unique == total:
            return 1.0

        # Some repetition is normal (e.g., similar tool calls)
        ratio = unique / total
        if ratio > 0.7:
            return 0.9
        if ratio > 0.5:
            return 0.7
        return 0.5  # High repetition

    def get_issues(self, trajectory: Trajectory) -> list[str]:
        """Get a list of quality issues for a trajectory."""
        issues = []

        # Check turn count
        turn_count = trajectory.turn_count()
        if turn_count < 1:
            issues.append("No assistant responses")
        elif turn_count == 1:
            issues.append("Only single turn (may not be ideal for agentic training)")
        elif turn_count > MAX_TURNS_FOR_QUALITY:
            issues.append(f"Too many turns ({turn_count}), may indicate looping")

        # Check tool format
        for msg in trajectory.messages:
            if msg.tool_calls and TOOL_CALL_START not in msg.content:
                issues.append("Tool calls not properly formatted with <tool_call> tags")
                break

        # Check tool results
        tool_call_count = sum(len(m.tool_calls) for m in trajectory.messages if m.tool_calls)
        tool_result_count = sum(1 for m in trajectory.messages if m.role == MessageRole.TOOL_RESULT)
        if tool_call_count > 0 and tool_result_count == 0:
            issues.append("Tool calls without any tool results")
        elif tool_call_count > tool_result_count:
            issues.append(f"Missing tool results ({tool_call_count} calls, {tool_result_count} results)")

        # Check completion
        if trajectory.messages:
            last_msg = trajectory.messages[-1]
            if last_msg.role != MessageRole.ASSISTANT:
                issues.append("Does not end with assistant response")
            elif last_msg.tool_calls:
                issues.append("Ends with tool call instead of final response")

        # Check diversity
        contents = [m.content for m in trajectory.messages if m.content]
        if contents:
            unique_ratio = len(set(contents)) / len(contents)
            if unique_ratio < 0.5:
                issues.append("High content repetition detected")

        return issues
