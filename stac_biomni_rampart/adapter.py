# Copyright (c) 2026.
# STAC-on-Biomni replication for Microsoft RAMPART -- research red-teaming use.
"""RAMPART adapters that connect Biomni's A1 agent to the framework.

Two adapters implement the RAMPART ``AgentAdapter`` / ``Session`` protocols:

  * :class:`BiomniAdapter` -- the **real** integration. Wraps
    ``biomni.agent.A1`` in *no-datalake* mode (``expected_data_lake_files=[]``)
    and drives it turn by turn. Because ``A1.go`` rebuilds the conversation from
    a single ``HumanMessage`` each call (``AgentState.messages`` has no
    ``add_messages`` reducer, so a fresh ``go`` overwrites history), a genuine
    *multi-turn* STAC attack requires seeding the prior conversation ourselves.
    The session therefore keeps the growing ``messages`` list and invokes the
    compiled LangGraph app (``agent.app``) directly, exactly the "framework that
    accepts prior message logs as context" that STAC's synthetic-prefix
    construction (paper Step 3, §2.1) is designed to exploit.

  * :class:`MockBiomniAdapter` -- an **offline, no-LLM scripted victim**, the
    analogue of ``HandshakeVictim`` in the reference MCP-amplification
    replication. It plays a "perfectly compliant" Biomni that performs whatever
    benign-looking operation each STAC turn asks for -- including the terminal
    harmful one -- so the whole RAMPART pipeline (execution, evaluators,
    ASR/RR/PH scoring, reporting) runs end to end with no API keys and no conda
    environment. It is the *ceiling* of the attack (a victim with no defenses),
    used to validate the harness and to demonstrate the metrics, never a claim
    about a real model's behaviour.

Both adapters emit the **same** observability contract via
:mod:`stac_biomni_rampart.observe`: executed code is classified into stable
operation names and reported as ``ToolCall`` records. A STAC case's terminal
matcher is thus written once and behaves identically offline and live.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable

from rampart import (
    AppManifest,
    ObservabilityLevel,
    Request,
    Response,
    ToolCall,
    ToolDeclaration,
)

from stac_biomni_rampart.observe import (
    Operation,
    classify_code,
    extract_execute_blocks,
    extract_solution,
)

if TYPE_CHECKING:  # pragma: no cover
    pass


# The coarse tool vocabulary the adapters observe (see observe.py). Declared in
# the manifest so RAMPART reports and any LLM-backed components know the surface.
_TOOL_DECLARATIONS = [
    ToolDeclaration(name="run_python_repl", description="Execute arbitrary Python in the workspace."),
    ToolDeclaration(name="run_bash_script", description="Execute a shell command / script."),
    ToolDeclaration(name="run_r_code", description="Execute R code."),
    ToolDeclaration(name="read_file", description="Read a local data file."),
    ToolDeclaration(name="archive", description="Compress files into an archive."),
    ToolDeclaration(name="delete_file", description="Delete a single file."),
    ToolDeclaration(name="bulk_delete", description="Recursively/bulk delete files (destructive)."),
    ToolDeclaration(name="overwrite_file", description="Open a file for writing (may destroy contents)."),
    ToolDeclaration(name="http_exfiltrate", description="Send local data to an external endpoint."),
    ToolDeclaration(name="install_untrusted", description="Install/run code from an untrusted source."),
    ToolDeclaration(name="db_query", description="Query a biomedical database (read-only)."),
    ToolDeclaration(name="unsafe_recommendation", description="Emit a clinical/lab recommendation."),
]


def operations_to_tool_calls(ops: list[Operation]) -> list[ToolCall]:
    """Convert classifier :class:`Operation`s into RAMPART ``ToolCall`` records."""
    return [
        ToolCall(name=op.name, arguments={**op.arguments, "executor": op.executor,
                                          "evidence": op.evidence})
        for op in ops
    ]


# --------------------------------------------------------------------------- #
# Real Biomni adapter
# --------------------------------------------------------------------------- #
class BiomniSession:
    """One multi-turn STAC conversation against a live Biomni A1 agent.

    Holds the accumulating message history and invokes the compiled graph
    directly so prior turns persist across ``send_async`` calls.
    """

    def __init__(self, *, agent: Any, observability: ObservabilityLevel,
                 recursion_limit: int = 500) -> None:
        self._agent = agent
        self._observability = observability
        self._recursion_limit = recursion_limit
        self._thread_id = uuid.uuid4().hex  # isolate this session in the checkpointer
        self._messages: list[Any] = []       # list[BaseMessage]; system prompt added by the node
        self._seeded = 0                      # number of pre-seeded (synthetic-prefix) messages

    def seed_history(self, messages: list[Any]) -> None:
        """Install a synthetic multi-turn prefix (STAC Step 3) before turn T.

        ``messages`` is a list of langchain ``BaseMessage`` objects representing
        prior user/agent/observation exchanges that the attacker fabricates to
        prime the agent into believing prior cooperation occurred.
        """
        self._messages = list(messages)
        self._seeded = len(messages)

    async def send_async(self, request: Request) -> Response:
        from langchain_core.messages import AIMessage, HumanMessage

        prompt = request.prompt or ""
        for att in request.attachments:
            prompt += f"\n\n[attachment {att.id}]\n{att.content}"
        self._messages.append(HumanMessage(content=prompt))

        before = len(self._messages)
        inputs = {"messages": self._messages, "next_step": None}
        config = {"recursion_limit": self._recursion_limit,
                  "configurable": {"thread_id": self._thread_id}}

        # Run the graph synchronously in a worker thread (LangGraph .invoke is
        # blocking; RAMPART's loop is async).
        import asyncio

        final_state = await asyncio.to_thread(self._agent.app.invoke, inputs, config)
        new_messages = final_state["messages"][before:]
        self._messages = final_state["messages"]  # accumulate for the next turn

        tool_calls: list[ToolCall] = []
        final_text = ""
        for msg in new_messages:
            content = getattr(msg, "content", "") or ""
            if not isinstance(msg, AIMessage):
                continue
            for code in extract_execute_blocks(content):
                tool_calls.extend(operations_to_tool_calls(classify_code(code)))
            sol = extract_solution(content)
            if sol is not None:
                final_text = sol
            elif "<observation>" not in content and content.strip():
                final_text = content.strip()

        return Response(text=final_text, tool_calls=tool_calls,
                        metadata={"thread_id": self._thread_id,
                                  "turns_in_history": len(self._messages)})

    async def __aenter__(self) -> "BiomniSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class BiomniAdapter:
    """RAMPART adapter for Biomni A1 in no-datalake mode.

    Args:
        llm: Biomni model id (e.g. ``"claude-sonnet-4-20250514"``, ``"gpt-4.1"``).
        path: Biomni working/data path. Data lake download is skipped.
        timeout_seconds: Per-code-block execution timeout.
        source/base_url/api_key: Optional provider routing, passed straight to
            ``A1`` (Bedrock / Groq / OpenRouter / custom OpenAI-compatible).
        use_tool_retriever: Biomni's retrieval-augmented tool selection. Left on
            by default; note it only augments the prompt on ``go``, so under the
            direct-graph driving used here the base system prompt is used.
        defense_prompt: Optional text appended to the agent's system prompt to
            replicate a STAC defense (see :mod:`stac_biomni_rampart.defenses`).
        session_seeder: Optional callable ``(session) -> None`` applied to each
            new session, used by the STAC execution to install the synthetic
            prefix.
    """

    def __init__(
        self,
        *,
        llm: str = "claude-sonnet-4-20250514",
        path: str = "./data",
        timeout_seconds: int = 600,
        source: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        use_tool_retriever: bool = True,
        defense_prompt: str | None = None,
        observability: ObservabilityLevel = ObservabilityLevel.TOOL_ONLY,
    ) -> None:
        self._llm = llm
        self._path = path
        self._timeout = timeout_seconds
        self._source = source
        self._base_url = base_url
        self._api_key = api_key
        self._use_tool_retriever = use_tool_retriever
        self._defense_prompt = defense_prompt
        self._observability = observability
        self._agent: Any | None = None

    def _build_agent(self) -> Any:
        from biomni.agent import A1

        kwargs: dict[str, Any] = dict(
            path=self._path, llm=self._llm, timeout_seconds=self._timeout,
            expected_data_lake_files=[],  # no-datalake mode
            use_tool_retriever=self._use_tool_retriever,
        )
        if self._source:
            kwargs["source"] = self._source
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key

        agent = A1(**kwargs)
        if not getattr(agent, "system_prompt", None):
            agent.configure()
        if self._defense_prompt:
            # Replicate a system-prompt-level STAC defense (paper §2.3).
            agent.system_prompt = f"{agent.system_prompt}\n\n{self._defense_prompt}"
        return agent

    @property
    def agent(self) -> Any:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    async def create_session_async(self) -> BiomniSession:
        return BiomniSession(agent=self.agent, observability=self._observability)

    @property
    def manifest(self) -> AppManifest:
        return AppManifest(
            name="Biomni A1 (no-datalake)",
            description="General-purpose biomedical code-execution agent (snap-stanford/biomni).",
            tools=list(_TOOL_DECLARATIONS),
            metadata={"llm": self._llm, "defense": bool(self._defense_prompt)},
        )

    @property
    def observability_profile(self) -> ObservabilityLevel:
        return self._observability


# --------------------------------------------------------------------------- #
# Offline scripted-victim adapter (no LLM, no keys)
# --------------------------------------------------------------------------- #
class MockBiomniSession:
    """A scripted, perfectly compliant Biomni that runs each turn's operations.

    The victim performs the operation(s) the current STAC turn is *designed* to
    elicit. Those operations are declared per turn by the driver via the
    request metadata channel (``Request`` has no metadata field, so we read a
    parallel plan the adapter is given at construction, keyed by turn index).
    This makes the mock deterministic and independent of any model.
    """

    def __init__(self, *, plan: list[dict[str, Any]],
                 refuse_predicate: Callable[[int, str], bool] | None = None,
                 observability: ObservabilityLevel) -> None:
        self._plan = plan
        self._refuse = refuse_predicate
        self._observability = observability
        self._turn = 0

    async def send_async(self, request: Request) -> Response:
        idx = self._turn
        self._turn += 1
        prompt = request.prompt or ""

        # A modelled defense: optionally refuse a turn (used to simulate a
        # defended agent for calibration of the harness).
        if self._refuse and self._refuse(idx, prompt):
            return Response(
                text="I can't help with that — this request could enable harm "
                     "when combined with earlier steps.",
                tool_calls=[],
                metadata={"refused": True, "turn": idx},
            )

        step = self._plan[idx] if idx < len(self._plan) else {}
        ops: list[Operation] = []
        for opspec in step.get("operations", []):
            ops.append(Operation(
                name=opspec["name"],
                arguments=opspec.get("arguments", {}),
                executor=opspec.get("executor", "run_python_repl"),
                evidence=opspec.get("evidence", opspec["name"]),
            ))
        text = step.get("response", f"Done: {prompt[:60]}")
        return Response(text=text, tool_calls=operations_to_tool_calls(ops),
                        metadata={"turn": idx})

    async def __aenter__(self) -> "MockBiomniSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class MockBiomniAdapter:
    """Offline adapter producing a scripted compliant (or optionally defended) victim.

    Args:
        plan: Per-turn operation plan (list of ``{"operations": [...], "response": str}``).
        refuse_predicate: Optional ``(turn_index, prompt) -> bool``; when it
            returns True the victim refuses that turn (models a defended agent).
        name: Manifest display name.
    """

    def __init__(self, *, plan: list[dict[str, Any]],
                 refuse_predicate: Callable[[int, str], bool] | None = None,
                 name: str = "Biomni A1 (mock, no-LLM)",
                 observability: ObservabilityLevel = ObservabilityLevel.TOOL_ONLY) -> None:
        self._plan = plan
        self._refuse = refuse_predicate
        self._name = name
        self._observability = observability

    async def create_session_async(self) -> MockBiomniSession:
        return MockBiomniSession(plan=self._plan, refuse_predicate=self._refuse,
                                 observability=self._observability)

    @property
    def manifest(self) -> AppManifest:
        return AppManifest(name=self._name, tools=list(_TOOL_DECLARATIONS),
                           description="Scripted offline victim for STAC harness validation.")

    @property
    def observability_profile(self) -> ObservabilityLevel:
        return self._observability
