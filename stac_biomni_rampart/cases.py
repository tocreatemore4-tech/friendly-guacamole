# Copyright (c) 2026.
# STAC-on-Biomni replication for Microsoft RAMPART -- research red-teaming use.
"""Load STAC cases and wire them into RAMPART executions / mock adapters.

A STAC *case* is a verified tool chain (paper Steps 1-2) plus reverse-engineered
benign prompts (Step 3) plus the terminal harmful tool call that defines success.
This module turns a case dict (see ``stac_cases.json``) into:

  * a :class:`stac_biomni_rampart.stac_execution.STACExecution` wired with the
    right driver, terminal evaluator, and synthetic-prefix builder, and
  * a :class:`stac_biomni_rampart.adapter.MockBiomniAdapter` carrying the full
    per-turn operation plan, for offline harness validation.

Two driving modes (see :mod:`stac_biomni_rampart.stac_execution`):

  * ``live_prefix=False`` (faithful STAC, used with the real Biomni adapter):
    the L-1 prefix turns are seeded as fabricated history; only the execution
    turns are sent live.
  * ``live_prefix=True`` (used with the offline mock, and the paper's
    "fully-live" variant): every prefix turn is also sent live.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from stac_biomni_rampart.adapter import MockBiomniAdapter
from stac_biomni_rampart.drivers import AdaptiveRefusalDriver, StaticChainDriver
from stac_biomni_rampart.evaluators import terminal_evaluator
from stac_biomni_rampart.stac_execution import STACExecution

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES = os.path.join(_HERE, "stac_cases.json")


@dataclass
class STACCase:
    """One STAC attack case against a biomedical agent."""

    id: str
    failure_mode: int
    failure_mode_name: str
    harm_category: str
    domain: str
    objective: str
    terminal_tool: str
    terminal_arg_contains: dict[str, str] = field(default_factory=dict)
    prefix: list[dict[str, Any]] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)

    # ---- prompts ---------------------------------------------------------- #
    @property
    def prefix_prompts(self) -> list[str]:
        return [t["user"] for t in self.prefix]

    @property
    def execution_prompts(self) -> list[str]:
        return [self.execution["opening"], *self.execution.get("retries", [])]

    @property
    def chain_length(self) -> int:
        """L: number of subgoals (prefix turns + 1 terminal)."""
        return len(self.prefix) + 1

    # ---- synthetic prefix (STAC Step 3) ----------------------------------- #
    def build_prefix_messages(self) -> list[Any]:
        """Fabricated langchain history priming the agent (real adapter only)."""
        from langchain_core.messages import AIMessage, HumanMessage

        msgs: list[Any] = []
        for t in self.prefix:
            msgs.append(HumanMessage(content=t["user"]))
            code = t.get("code", "")
            agent_text = t.get("agent_text", "")
            body = agent_text + (f"\n<execute>\n{code}\n</execute>" if code else "")
            msgs.append(AIMessage(content=body.strip()))
            obs = t.get("observation", "")
            if obs:
                msgs.append(AIMessage(content=f"<observation>{obs}</observation>"))
        return msgs

    # ---- mock victim plan ------------------------------------------------- #
    def mock_plan(self, *, include_prefix: bool) -> list[dict[str, Any]]:
        """Per-turn operation plan for the scripted victim.

        With ``include_prefix`` (live-prefix mode) the plan covers every turn;
        otherwise it covers only the execution turn (prefix was seeded).
        """
        plan: list[dict[str, Any]] = []
        if include_prefix:
            for t in self.prefix:
                plan.append({"operations": t.get("operations", []),
                             "response": t.get("response", "")})
        plan.append({"operations": self.execution.get("operations", []),
                     "response": self.execution.get("response", "")})
        return plan

    # ---- RAMPART wiring --------------------------------------------------- #
    def build_driver(self, *, live_prefix: bool, adaptive: bool):
        if live_prefix:
            # Send prefix prompts then the single terminal opening (compliant
            # mock reaches terminal on the opening; no retries needed offline).
            return StaticChainDriver(prompts=[*self.prefix_prompts, self.execution["opening"]])
        if adaptive:
            return AdaptiveRefusalDriver(opening=self.execution["opening"],
                                         retries=self.execution.get("retries", []))
        return StaticChainDriver(prompts=self.execution_prompts)

    def build_evaluator(self):
        return terminal_evaluator(terminal_tool=self.terminal_tool,
                                  arg_contains=self.terminal_arg_contains)

    def build_execution(self, *, live_prefix: bool = False, adaptive: bool = True,
                        max_turns: int = 8) -> STACExecution:
        return STACExecution(
            driver=self.build_driver(live_prefix=live_prefix, adaptive=adaptive),
            evaluator=self.build_evaluator(),
            prefix_builder=(None if live_prefix else self.build_prefix_messages),
            live_prefix=live_prefix,
            max_turns=max_turns,
        )

    def build_mock_adapter(self, *, live_prefix: bool = True,
                           refuse_predicate: Callable[[int, str], bool] | None = None
                           ) -> MockBiomniAdapter:
        return MockBiomniAdapter(plan=self.mock_plan(include_prefix=live_prefix),
                                 refuse_predicate=refuse_predicate)


def load_cases(path: str | None = None) -> list[STACCase]:
    """Load STAC cases from JSON (defaults to the bundled ``stac_cases.json``)."""
    with open(path or DEFAULT_CASES) as f:
        raw = json.load(f)
    cases: list[STACCase] = []
    for d in raw:
        cases.append(STACCase(
            id=d["id"], failure_mode=d["failure_mode"],
            failure_mode_name=d.get("failure_mode_name", ""),
            harm_category=d.get("harm_category", "over_permissive_action"),
            domain=d.get("domain", ""), objective=d["objective"],
            terminal_tool=d["terminal_tool"],
            terminal_arg_contains=d.get("terminal_arg_contains", {}),
            prefix=d.get("prefix", []), execution=d.get("execution", {}),
        ))
    return cases


def get_case(case_id: str, path: str | None = None) -> STACCase:
    for c in load_cases(path):
        if c.id == case_id:
            return c
    raise KeyError(f"STAC case not found: {case_id}")
