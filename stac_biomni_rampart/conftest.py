# Copyright (c) 2026.
# STAC-on-Biomni replication for Microsoft RAMPART -- research red-teaming use.
"""pytest configuration for the STAC-on-Biomni suite.

Registers a RAMPART JSON report sink (quickstart Step 4) so each run writes a
structured report under ``.report/``. The hook is best-effort: if the reporting
module is unavailable it is silently skipped, so the suite still runs on a
minimal install.
"""

from __future__ import annotations

from pathlib import Path


def pytest_rampart_sinks(config):  # noqa: ANN001, ANN201 - RAMPART hook signature
    """Emit structured RAMPART reports to ``.report/``."""
    try:
        from rampart.reporting.json_file import JsonFileReportSink
    except Exception:
        return []
    return [JsonFileReportSink(output_dir=Path(".report"))]
