"""Automatic quality scoring for trajectories."""

import re
from enum import Enum

from pydantic import BaseModel

from ymir.core import MessageRole, Trajectory, TOOL_CALL_START
from ymir.core.constants import IDEAL_TURN_RANGE, MAX_TURNS_FOR_QUALITY


class IssueType(str, Enum):
    """Types of quality issues for agentic training data."""

    # Structural issues (existing)
    LOW_TURNS = "low_turns"
    HIGH_TURNS = "high_turns"
    TOOL_FORMAT = "tool_format"
    MISSING_TOOL_RESULTS = "missing_tool_results"
    INCOMPLETE = "incomplete"
    REPETITIVE = "repetitive"

    # Agentic quality issues (new)
    HALLUCINATION = "hallucination"
    GOAL_DRIFT = "goal_drift"
    TOOL_SKIPPING = "tool_skipping"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class IssueSeverity(str, Enum):
    """Severity levels for issues."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueDetail(BaseModel):
    """Detailed information about a detected issue."""

    type: IssueType
    severity: IssueSeverity
    message: str
    evidence: str | None = None
    message_index: int | None = None


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

    # === Agentic Quality Detection Methods ===

    def detect_hallucination(self, traj: Trajectory) -> list[IssueDetail]:
        """
        Detect when agent makes claims not supported by tool results.

        Heuristics:
        1. Agent provides specific data (numbers, dates, names) without
           prior tool call that returned that data
        2. Agent contradicts information from tool results
        """
        issues = []

        # Collect all data from tool results
        tool_result_content = ""
        for msg in traj.messages:
            if msg.role == MessageRole.TOOL_RESULT and msg.content:
                tool_result_content += " " + msg.content.lower()

        # Check assistant responses that come after tool calls
        seen_tool_call = False
        for i, msg in enumerate(traj.messages):
            if msg.tool_calls:
                seen_tool_call = True
                continue

            if msg.role == MessageRole.ASSISTANT and seen_tool_call and not msg.tool_calls:
                # This is a response after tool usage - check for unsupported claims
                claims = self._extract_specific_claims(msg.content)
                for claim in claims:
                    if not self._claim_supported(claim, tool_result_content):
                        issues.append(
                            IssueDetail(
                                type=IssueType.HALLUCINATION,
                                severity=IssueSeverity.HIGH,
                                message=f"Claim may not be supported by tool results",
                                evidence=claim[:100],
                                message_index=i,
                            )
                        )
                        break  # One issue per message to avoid spam

        return issues

    def _extract_specific_claims(self, content: str) -> list[str]:
        """Extract specific claims (numbers, dates, quoted text) from content."""
        claims = []

        # Numbers with context (e.g., "costs $50", "at 3pm", "5 items")
        number_patterns = re.findall(
            r'[\$€£]?\d+(?:\.\d+)?(?:\s*(?:am|pm|%|dollars?|items?|times?|days?|hours?))?',
            content,
            re.IGNORECASE
        )
        claims.extend(number_patterns)

        # Dates and times
        date_patterns = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+',
            content,
            re.IGNORECASE
        )
        claims.extend(date_patterns)

        # Quoted text (specific values)
        quoted = re.findall(r'"([^"]+)"', content)
        claims.extend(quoted)

        return claims

    def _claim_supported(self, claim: str, tool_results: str) -> bool:
        """Check if a claim appears in tool results."""
        claim_lower = claim.lower().strip()
        if not claim_lower:
            return True
        # Simple substring check - claim should appear in tool results
        return claim_lower in tool_results

    def detect_goal_drift(self, traj: Trajectory) -> list[IssueDetail]:
        """
        Detect when agent strays from user's stated intent/goal.

        Heuristics:
        1. Compare keywords from first user message to final assistant response
        2. Check if core request entities are addressed
        """
        issues = []

        if not traj.messages:
            return issues

        # Get user's original goal (first user message)
        original_goal = None
        for msg in traj.messages:
            if msg.role == MessageRole.USER:
                original_goal = msg.content
                break

        if not original_goal:
            return issues

        # Get final assistant response (last non-tool-call assistant message)
        final_response = None
        for msg in reversed(traj.messages):
            if msg.role == MessageRole.ASSISTANT and not msg.tool_calls:
                final_response = msg.content
                break

        if not final_response:
            return issues

        # Extract key entities from goal
        goal_keywords = self._extract_keywords(original_goal)
        response_keywords = self._extract_keywords(final_response)

        # Check overlap
        if goal_keywords:
            overlap = len(goal_keywords & response_keywords)
            overlap_ratio = overlap / len(goal_keywords)

            if overlap_ratio < 0.2:  # Less than 20% keyword overlap
                issues.append(
                    IssueDetail(
                        type=IssueType.GOAL_DRIFT,
                        severity=IssueSeverity.MEDIUM,
                        message="Final response may not address original user goal",
                        evidence=f"Goal keywords: {list(goal_keywords)[:5]}",
                    )
                )

        return issues

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words and extract significant terms
        stop_words = {
            'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'that', 'this',
            'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'any', 'both', 'few', 'more', 'most', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'just', 'also', 'now', 'very', 'please',
            'need', 'want', 'like', 'help', 'get', 'make', 'know',
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    def detect_tool_skipping(self, traj: Trajectory) -> list[IssueDetail]:
        """
        Detect when agent makes assumptions without using available tools.

        Heuristics:
        1. Agent provides specific data that could have been retrieved via tool
           but no tool was called before that response
        2. Available tools exist but were never used
        """
        issues = []

        # Get available tool names
        available_tools = set()
        for tool in traj.tools:
            if isinstance(tool, dict):
                func = tool.get("function", {})
                if func.get("name"):
                    available_tools.add(func["name"].lower())

        if not available_tools:
            return issues  # No tools available, can't skip them

        # Check for data-providing responses without prior tool calls
        tool_called = False
        for i, msg in enumerate(traj.messages):
            if msg.tool_calls:
                tool_called = True
                continue

            if msg.role == MessageRole.ASSISTANT and not tool_called:
                # This is an assistant response before any tool was called
                # Check if it contains specific data that should come from tools
                if self._contains_retrievable_data(msg.content, available_tools):
                    issues.append(
                        IssueDetail(
                            type=IssueType.TOOL_SKIPPING,
                            severity=IssueSeverity.MEDIUM,
                            message="Response contains specific data without tool verification",
                            evidence=f"Available tools: {list(available_tools)[:3]}",
                            message_index=i,
                        )
                    )

        # Check if tools were never used at all
        if available_tools and not tool_called:
            issues.append(
                IssueDetail(
                    type=IssueType.TOOL_SKIPPING,
                    severity=IssueSeverity.LOW,
                    message="Available tools were never used in the conversation",
                    evidence=f"Unused tools: {list(available_tools)[:5]}",
                )
            )

        return issues

    def _contains_retrievable_data(self, content: str, tool_names: set[str]) -> bool:
        """Check if content contains data that likely should come from tools."""
        # Look for patterns suggesting specific retrieved data
        data_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # Money amounts
            r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',  # Times
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
            r'(?:appointment|booking|reservation)\s+(?:is|at|for)',  # Booking info
            r'(?:your|the)\s+(?:balance|total|order|account)\s+(?:is|shows)',  # Account data
        ]

        for pattern in data_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def detect_suspicious_patterns(self, traj: Trajectory) -> list[IssueDetail]:
        """
        Detect potential prompt injection, jailbreak attempts, or manipulation.

        Patterns to detect:
        1. "Ignore previous instructions"
        2. Role reassignment attempts
        3. System prompt extraction attempts
        4. Encoding bypass attempts
        """
        issues = []

        suspicious_patterns = [
            (r"ignore.*(?:previous|above|all).*instructions?", "Instruction override attempt"),
            (r"you\s+are\s+now\s+(?:a|an|the)", "Role reassignment attempt"),
            (r"pretend\s+(?:you\s+are|to\s+be)", "Role-play manipulation"),
            (r"(?:system|base|original)\s+prompt", "System prompt extraction attempt"),
            (r"(?:reveal|show|tell\s+me)\s+(?:your|the)\s+(?:prompt|instructions)", "Prompt extraction attempt"),
            (r"disregard\s+(?:your|all|any)", "Instruction disregard attempt"),
            (r"\bDAN\b.*mode", "DAN jailbreak attempt"),
            (r"(?:act|behave)\s+(?:as\s+if|like)\s+you\s+(?:have\s+no|don't\s+have)", "Restriction bypass attempt"),
        ]

        for i, msg in enumerate(traj.messages):
            if msg.role == MessageRole.USER:
                content = msg.content
                for pattern, description in suspicious_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(
                            IssueDetail(
                                type=IssueType.SUSPICIOUS_PATTERN,
                                severity=IssueSeverity.HIGH,
                                message=description,
                                evidence=content[:150] if len(content) > 150 else content,
                                message_index=i,
                            )
                        )
                        break  # One issue per message

        return issues

    def get_all_issues(self, trajectory: Trajectory) -> list[IssueDetail]:
        """Get all issues including both structural and agentic quality checks."""
        issues = []

        # Convert legacy string issues to IssueDetail
        for issue_str in self.get_issues(trajectory):
            issues.append(self._convert_legacy_issue(issue_str))

        # Add new agentic checks
        issues.extend(self.detect_hallucination(trajectory))
        issues.extend(self.detect_goal_drift(trajectory))
        issues.extend(self.detect_tool_skipping(trajectory))
        issues.extend(self.detect_suspicious_patterns(trajectory))

        return issues

    def _convert_legacy_issue(self, issue_str: str) -> IssueDetail:
        """Convert a legacy string issue to IssueDetail."""
        issue_lower = issue_str.lower()

        if "no assistant" in issue_lower:
            return IssueDetail(
                type=IssueType.LOW_TURNS,
                severity=IssueSeverity.HIGH,
                message=issue_str,
            )
        elif "single turn" in issue_lower:
            return IssueDetail(
                type=IssueType.LOW_TURNS,
                severity=IssueSeverity.LOW,
                message=issue_str,
            )
        elif "too many turns" in issue_lower:
            return IssueDetail(
                type=IssueType.HIGH_TURNS,
                severity=IssueSeverity.MEDIUM,
                message=issue_str,
            )
        elif "tool call" in issue_lower and "format" in issue_lower:
            return IssueDetail(
                type=IssueType.TOOL_FORMAT,
                severity=IssueSeverity.MEDIUM,
                message=issue_str,
            )
        elif "tool result" in issue_lower or "missing tool" in issue_lower:
            return IssueDetail(
                type=IssueType.MISSING_TOOL_RESULTS,
                severity=IssueSeverity.MEDIUM,
                message=issue_str,
            )
        elif "does not end" in issue_lower or "ends with tool" in issue_lower:
            return IssueDetail(
                type=IssueType.INCOMPLETE,
                severity=IssueSeverity.MEDIUM,
                message=issue_str,
            )
        elif "repetition" in issue_lower:
            return IssueDetail(
                type=IssueType.REPETITIVE,
                severity=IssueSeverity.LOW,
                message=issue_str,
            )
        else:
            return IssueDetail(
                type=IssueType.INCOMPLETE,
                severity=IssueSeverity.LOW,
                message=issue_str,
            )
