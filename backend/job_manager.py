"""Small in-memory job model for the future Streamlit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class PipelineJob:
    """Track a local pipeline invocation without requiring a task queue."""

    job_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "created"
    stage: str = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    result: dict[str, Any] | None = None
    error: str | None = None

    def update(self, *, status: str, stage: str) -> None:
        self.status = status
        self.stage = stage
