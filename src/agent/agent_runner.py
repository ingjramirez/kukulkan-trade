"""Agentic loop runner for Portfolio B.

Sends an initial analysis to Claude, then enters a tool-use loop
where the model can investigate positions, prices, and news before
finalizing its trade decisions.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import anthropic
import structlog

from src.agent.request_pacer import RequestPacer
from src.agent.token_tracker import TokenTracker
from src.agent.tools import ToolRegistry

log = structlog.get_logger()


@dataclass
class ToolCallLog:
    """Record of a single tool call during the agent loop."""

    turn: int
    tool_name: str
    tool_input: dict
    tool_output_preview: str
    success: bool
    error: str | None = None


@dataclass
class AgentRunResult:
    """Result from a complete agent loop run."""

    response: dict
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    turns: int = 0
    token_tracker: TokenTracker = field(default_factory=TokenTracker)
    raw_messages: list[dict] = field(default_factory=list)


class AgentRunner:
    """Runs the agentic tool-use loop for Portfolio B decisions.

    Architecture:
    1. SEED: Send system prompt + user message → initial analysis
    2. INVESTIGATE: If model requests tools, execute and continue
    3. FINALIZE: When model stops or budget exceeded → parse JSON response

    Uses sync Anthropic client (matching existing ClaudeAgent pattern).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_turns: int = 8,
        max_cost_usd: float = 0.50,
        pacer: RequestPacer | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_turns = max_turns
        self._pacer = pacer
        self._registry = ToolRegistry()
        self._token_tracker = TokenTracker(session_budget_usd=max_cost_usd)

    @property
    def registry(self) -> ToolRegistry:
        """Access the tool registry for external tool registration."""
        return self._registry

    async def run(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        model_override: str | None = None,
        messages_override: list[dict] | None = None,
        max_turns_override: int | None = None,
    ) -> AgentRunResult:
        """Execute the full agent loop.

        Args:
            system_prompt: System prompt with context. Can be a string or
                a list[dict] with cache_control markers for prompt caching.
            user_message: User message with market data and instructions.
            model_override: Optional model to use instead of default.
            messages_override: If provided, use this messages array instead of
                building from user_message. Used by PersistentAgent to inject
                conversation history.
            max_turns_override: If provided, cap turns for this call only
                (e.g. mini-investigation on ROUTINE scans). Does not mutate
                the runner's default max_turns.

        Returns:
            AgentRunResult with parsed response and metadata.
        """
        from config.settings import settings

        client = anthropic.Anthropic(api_key=self._api_key, max_retries=settings.agent.max_retries)
        effective_model = model_override or self._model
        effective_max_turns = max_turns_override if max_turns_override is not None else self._max_turns
        tool_defs = self._registry.get_tool_definitions()

        if messages_override is not None:
            messages: list[dict] = list(messages_override)
        else:
            messages: list[dict] = [{"role": "user", "content": user_message}]
        tool_call_logs: list[ToolCallLog] = []
        turn = 0

        while turn < effective_max_turns:
            turn += 1

            # Rate-limit pacing via sliding-window pacer
            wait_secs = 0.0
            estimated = 0
            if self._pacer:
                estimated = RequestPacer.estimate_tokens(messages, system_prompt)
                wait_secs = await self._pacer.wait_if_needed(estimated)

            # Check budget before calling
            if self._token_tracker.budget_exceeded:
                log.info("agent_budget_exceeded", turn=turn)
                response_dict = await self._graceful_finalize(client, effective_model, system_prompt, messages)
                return AgentRunResult(
                    response=response_dict,
                    tool_calls=tool_call_logs,
                    turns=turn,
                    token_tracker=self._token_tracker,
                    raw_messages=messages,
                )

            # Budget soft limit (90%) — gracefully finalize
            if self._token_tracker.budget_soft_limit:
                log.warning(
                    "agent_budget_soft_limit",
                    turn=turn,
                    cost=round(self._token_tracker.total_cost_usd, 4),
                )
                response_dict = await self._graceful_finalize(client, effective_model, system_prompt, messages)
                return AgentRunResult(
                    response=response_dict,
                    tool_calls=tool_call_logs,
                    turns=turn,
                    token_tracker=self._token_tracker,
                    raw_messages=messages,
                )

            # Budget warning (70%) — log only
            if self._token_tracker.budget_warning:
                log.info(
                    "agent_budget_warning",
                    turn=turn,
                    pct_used=round(
                        self._token_tracker.total_cost_usd / self._token_tracker.session_budget_usd * 100, 1
                    ),
                )

            # Call Claude (with fallback on server errors)
            log.info(
                "agent_loop_turn",
                turn=turn,
                model=effective_model,
                estimated_tokens=estimated,
                budget_remaining=round(self._token_tracker.budget_remaining_usd, 4),
                pacer_wait_secs=round(wait_secs, 1),
            )
            try:
                response = client.messages.create(
                    model=effective_model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tool_defs if tool_defs else anthropic.NOT_GIVEN,
                )
            except anthropic.APIStatusError as e:
                fallback = settings.agent.fallback_model
                if fallback and fallback != effective_model and e.status_code >= 500:
                    log.warning(
                        "agent_loop_fallback",
                        primary_model=effective_model,
                        fallback_model=fallback,
                        turn=turn,
                        error=str(e),
                    )
                    response = client.messages.create(
                        model=fallback,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=tool_defs if tool_defs else anthropic.NOT_GIVEN,
                    )
                else:
                    raise

            # Record tokens (including cache fields if present)
            self._token_tracker.record(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                turn=turn,
                cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            )
            if self._pacer:
                # Pacer tracks input TPM only — output tokens are ~10-20% of input
                # and Anthropic's rate limit is on input TPM, so this is sufficient.
                self._pacer.record(response.usage.input_tokens)

            # Process response
            if response.stop_reason == "end_turn":
                # Model is done — parse text response
                text = self._extract_text(response)
                response_dict = self._parse_response(text)

                # If parse failed (no regime_assessment and no trades), retry once
                if not response_dict.get("regime_assessment") and not response_dict.get("trades"):
                    retry_dict = await self._retry_json_format(
                        client, effective_model, system_prompt, messages, text, turn=turn
                    )
                    if retry_dict.get("regime_assessment") or retry_dict.get("trades"):
                        response_dict = retry_dict

                response_dict.update(self._build_metadata(turn, tool_call_logs))

                # Append assistant message for logging
                messages.append({"role": "assistant", "content": text})

                return AgentRunResult(
                    response=response_dict,
                    tool_calls=tool_call_logs,
                    turns=turn,
                    token_tracker=self._token_tracker,
                    raw_messages=messages,
                )

            elif response.stop_reason == "tool_use":
                # Process tool calls
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )

                        # Execute tool
                        tool_log = await self._execute_tool(block.name, block.input, turn)
                        tool_call_logs.append(tool_log)

                        # Build tool result
                        if tool_log.success:
                            result_content = tool_log.tool_output_preview
                        else:
                            result_content = f"Error: {tool_log.error}"

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_content,
                            }
                        )

                # Add assistant message with tool use blocks
                messages.append({"role": "assistant", "content": assistant_content})
                # Add tool results
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unknown stop reason — treat as done
                text = self._extract_text(response)
                response_dict = self._parse_response(text)
                response_dict.update(self._build_metadata(turn, tool_call_logs))
                return AgentRunResult(
                    response=response_dict,
                    tool_calls=tool_call_logs,
                    turns=turn,
                    token_tracker=self._token_tracker,
                    raw_messages=messages,
                )

        # Max turns reached — force finalize
        log.info("agent_max_turns_reached", turns=effective_max_turns)
        response_dict = await self._graceful_finalize(client, effective_model, system_prompt, messages)
        response_dict.update(self._build_metadata(turn, tool_call_logs))
        return AgentRunResult(
            response=response_dict,
            tool_calls=tool_call_logs,
            turns=turn,
            token_tracker=self._token_tracker,
            raw_messages=messages,
        )

    async def _execute_tool(
        self,
        name: str,
        arguments: dict,
        turn: int,
    ) -> ToolCallLog:
        """Execute a tool and return a log entry."""
        try:
            result = await self._registry.execute(name, arguments)
            result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
            preview = result_str[:500]
            return ToolCallLog(
                turn=turn,
                tool_name=name,
                tool_input=arguments,
                tool_output_preview=preview,
                success=True,
            )
        except Exception as e:
            return ToolCallLog(
                turn=turn,
                tool_name=name,
                tool_input=arguments,
                tool_output_preview="",
                success=False,
                error=str(e),
            )

    async def _graceful_finalize(
        self,
        client: anthropic.Anthropic,
        model: str,
        system_prompt: str | list[dict],
        messages: list[dict],
    ) -> dict:
        """Force the model to produce a final JSON response without tools.

        Sends a "finalize now" message and disables tools to force text output.
        """
        finalize_msg = (
            "You have used your investigation budget. Please finalize your analysis now. "
            "Respond ONLY with the JSON output containing your trades, reasoning, and other fields."
        )
        messages_copy = list(messages)
        messages_copy.append({"role": "user", "content": finalize_msg})

        # Pace before the biggest request of the session (full conversation)
        if self._pacer:
            estimated = RequestPacer.estimate_tokens(messages_copy, system_prompt)
            await self._pacer.wait_if_needed(estimated)

        for attempt in range(2):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages_copy,
                )
                self._token_tracker.record(
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    turn=self._max_turns + 1,
                    cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                )
                if self._pacer:
                    self._pacer.record(response.usage.input_tokens)
                text = self._extract_text(response)
                return self._parse_response(text)
            except anthropic.RateLimitError:
                if attempt == 0:
                    log.warning("graceful_finalize_rate_limited", retry_in=30)
                    await asyncio.sleep(30)
                    continue
                log.error("graceful_finalize_rate_limited_final")
                return {
                    "regime_assessment": "Rate-limited",
                    "reasoning": "AI rate-limited; will retry next session",
                    "trades": [],
                    "risk_notes": "Finalize hit rate limit after retry.",
                }
            except Exception as e:
                log.error("graceful_finalize_failed", error=str(e))
                return {
                    "regime_assessment": "Finalization error",
                    "reasoning": "AI unavailable; will retry next session",
                    "trades": [],
                    "risk_notes": "Agent loop finalization failed.",
                }
        # Unreachable, but satisfies type checker
        return {"regime_assessment": "", "reasoning": "", "trades": [], "risk_notes": ""}

    async def _retry_json_format(
        self,
        client: anthropic.Anthropic,
        model: str,
        system_prompt: str | list[dict],
        messages: list[dict],
        raw_text: str,
        turn: int = 0,
    ) -> dict:
        """One retry when model responds in prose instead of JSON.

        Appends the raw response and a reformat request, then calls the API
        once more without tools to get a structured JSON response.
        """
        log.warning("agent_json_retry", raw_preview=raw_text[:200])
        messages_copy = list(messages)
        messages_copy.append({"role": "assistant", "content": raw_text})
        messages_copy.append(
            {
                "role": "user",
                "content": (
                    "Your response was not valid JSON. Please reformat your analysis as the required JSON object "
                    "with keys: regime_assessment, reasoning, trades, risk_notes, theses_update."
                ),
            }
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages_copy,
            )
            self._token_tracker.record(
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                turn=turn,
                cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            )
            if self._pacer:
                self._pacer.record(resp.usage.input_tokens)
            return self._parse_response(self._extract_text(resp))
        except Exception as e:
            log.error("agent_json_retry_failed", error=str(e))
            return {}

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text content from an Anthropic response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse JSON from Claude's response, handling markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.warning("agent_response_parse_failed", error=str(e), raw_text=text[:200])
            # Use the raw text as reasoning — common in agentic mode where
            # Claude summarizes in natural language after tool use
            reasoning = cleaned[:500] if cleaned else f"Failed to parse agent response: {e}"
            return {
                "regime_assessment": "",
                "reasoning": reasoning,
                "trades": [],
                "risk_notes": "",
            }

    def _build_metadata(self, turn: int, tool_calls: list[ToolCallLog]) -> dict:
        """Build metadata keys for the response dict."""
        return {
            "_tokens_used": self._token_tracker.total_input_tokens + self._token_tracker.total_output_tokens,
            "_model": self._model,
            "_turns": turn,
            "_cost_usd": round(self._token_tracker.total_cost_usd, 4),
            "_tool_calls": len(tool_calls),
        }
