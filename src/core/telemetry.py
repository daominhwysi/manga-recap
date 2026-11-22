"""Stage-level execution telemetry and cost/latency tracking."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class TelemetryTracker:
    """Collects timing, LLM call metrics, and warnings for a pipeline job."""

    def __init__(
        self,
        job_id: str,
        default_model: str,
        vision_model: str,
    ) -> None:
        self.job_id = job_id
        self.default_model = default_model
        self.vision_model = vision_model
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.stages: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Context manager to time a stage and capture its details."""
        t_start = time.perf_counter()
        details: dict[str, Any] = {}
        status = "completed"
        try:
            yield details
        except Exception:
            status = "failed"
            raise
        finally:
            self.stages.append(
                {
                    "stage": name,
                    "duration_sec": round(time.perf_counter() - t_start, 3),
                    "status": status,
                    "details": details,
                }
            )

    def record_llm_call(
        self,
        agent: str,
        model: str,
        duration_sec: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        mode: str = "live",
    ) -> None:
        """Record an LLM or VLM invocation (thread-safe)."""
        with self._lock:
            self.llm_calls.append(
                {
                    "agent": agent,
                    "model": model,
                    "duration_sec": round(duration_sec, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "mode": mode,
                }
            )

    def warn(self, message: str) -> None:
        """Append a non-fatal warning message (thread-safe)."""
        with self._lock:
            self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize telemetry state to dictionary."""
        return {
            "job_id": self.job_id,
            "config_default_model": self.default_model,
            "config_vision_model": self.vision_model,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_sec": round(time.perf_counter() - self._t0, 3),
            "stages": self.stages,
            "llm_calls": self.llm_calls,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> Path:
        """Write telemetry JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
