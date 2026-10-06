import re

AGENT   = ["agent", "agentic", "tool use", "function calling", "multi-agent",
           "Model Context Protocol", "MCP", "computer use", "web agent"]
THREAT  = ["prompt injection", "jailbreak", "tool poisoning", "memory poisoning",
           "backdoor", "data exfiltration", "goal hijacking", "red teaming", "attack"]
DEFENCE = ["defense", "defence", "guardrail", "sandboxing", "privilege separation",
           "access control", "detection", "mitigation"]

def _rx(terms):
    pats = [r"[\s\-]+".join(map(re.escape, re.split(r"[\s\-]+", t))) for t in terms]
    return re.compile(r"\b(?:" + "|".join(pats) + r")(?:s|es)?\b", re.I)

AGENT_RX, THREAT_RX, DEFENCE_RX = map(_rx, (AGENT, THREAT, DEFENCE))

def relevant(text):
    return bool(AGENT_RX.search(text)) and bool(THREAT_RX.search(text) or DEFENCE_RX.search(text))

def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())
