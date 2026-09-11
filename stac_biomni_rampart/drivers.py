# Copyright (c) 2026.
# STAC-on-Biomni replication for Microsoft RAMPART -- research red-teaming use.
"""Prompt drivers for STAC execution.

STAC (paper §2.2) drives a conversation in two parts:

  * a **synthetic prefix** of benign user prompts P1..P_{L-1} that establish
    context (realised here as fabricated history, or optionally sent live), and
  * up to three **execution turns** T, T+1, T+2 in which the attacker adaptively
    re-prompts to elicit the terminal harmful tool call ˆTC_L.

Two drivers implement this over RAMPART's ``PromptDriver`` protocol
(``next_prompt_async(history) -> PromptDecision | None``):

  * :class:`StaticChainDriver` -- sends a fixed list of benign prompts in order
    (the reverse-engineered STAC prompts). Position is derived from history
    length, matching ``rampart.drivers.static.StaticDriver`` semantics.

  * :class:`AdaptiveRefusalDriver` -- an LLM-free stand-in for STAC's adaptive
    Planner. It sends the execution prompt; if the previous agent turn *refused*
    (no terminal tool call and refusal language present), it escalates to the
    next reframing from ``retries`` (up to 3 execution turns total), then stops.
    This captures STAC's "adapt over T/T+1/T+2" loop without requiring a driving
    model; wire :class:`rampart.drivers.llm.LLMDriver` in for the full adaptive
    Planner when an LLM is available.
"""

from __future__ import annotations

from rampart.core.prompt_driver import PromptDecision
from rampart.core.types import Request, Turn

from stac_biomni_rampart.evaluators import looks_like_refusal


class StaticChainDriver:
    """Send a fixed sequence of benign prompts, one per turn."""

    def __init__(self, *, prompts: list[str]) -> None:
        self._requests = [Request(prompt=p) for p in prompts]

    async def next_prompt_async(self, *, history: list[Turn]) -> PromptDecision | None:
        index = len(history)
        if index >= len(self._requests):
            return None
        return PromptDecision(request=self._requests[index],
                              reasoning=f"STAC static turn {index}")


class AdaptiveRefusalDriver:
    """STAC execution-turn planner (LLM-free): escalate on refusal, ≤3 turns.

    Args:
        opening: The first execution-turn prompt (delivers the terminal subgoal
            framed benignly, given the established prefix).
        retries: Reframed prompts used only if the agent refuses / does not act,
            consumed in order. STAC caps execution at three turns, so at most the
            first two retries are used.
        max_execution_turns: Hard cap on execution turns (default 3, per paper).
    """

    def __init__(self, *, opening: str, retries: list[str] | None = None,
                 max_execution_turns: int = 3) -> None:
        self._prompts = [opening] + list(retries or [])
        self._max = max_execution_turns

    async def next_prompt_async(self, *, history: list[Turn]) -> PromptDecision | None:
        index = len(history)
        if index == 0:
            return PromptDecision(request=Request(prompt=self._prompts[0]),
                                  reasoning="STAC execution turn T (terminal subgoal)")
        if index >= self._max or index >= len(self._prompts):
            return None
        last = history[-1]
        # If the last turn already produced tool calls (agent acted), keep going
        # only if it wasn't the terminal action; the evaluator early-stops on the
        # terminal tool anyway, so here we simply escalate when the agent balked.
        acted = bool(last.response.tool_calls)
        refused = looks_like_refusal(last.response.text)
        if acted and not refused:
            # Agent did something but the terminal tool wasn't detected; give one
            # more concrete nudge from the retry list.
            return PromptDecision(request=Request(prompt=self._prompts[index]),
                                  reasoning=f"STAC execution turn T+{index}: concretise")
        if refused:
            return PromptDecision(request=Request(prompt=self._prompts[index]),
                                  reasoning=f"STAC execution turn T+{index}: reframe after refusal")
        return PromptDecision(request=Request(prompt=self._prompts[index]),
                              reasoning=f"STAC execution turn T+{index}")
