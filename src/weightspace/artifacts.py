"""Unique run directories and append-only JSON records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunJournal:
    """Append events to one run. Create a fresh directory for each attempt."""

    def __init__(self, root: str | Path, metadata: dict, run_id: str | None = None):
        if run_id is None:
            run_id = uuid4().hex
        if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
            raise ValueError("invalid run ID")
        payload = canonical_json({
            "schema_version": 1,
            "run_id": run_id,
            "created_at": _timestamp(),
            "metadata_sha256": json_digest(metadata),
            "metadata": metadata,
        })
        path = Path(root) / run_id
        path.mkdir(parents=True, exist_ok=False)
        with (path / "manifest.json").open("xb") as file:
            file.write(payload + b"\n")
            file.flush()
            os.fsync(file.fileno())
        (path / "events.jsonl").touch(exist_ok=False)
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        self._sequence = 0

    @classmethod
    def create(cls, root: str | Path, metadata: dict, run_id: str | None = None) -> RunJournal:
        return cls(root, metadata, run_id)

    def append(self, event: str, **fields: Any) -> None:
        with self._lock:
            record = canonical_json({
                **fields, "event": event, "sequence": self._sequence,
                "timestamp": _timestamp(), "run_id": self.run_id,
            })
            with (self.path / "events.jsonl").open("ab") as file:
                file.write(record + b"\n")
                file.flush()
                os.fsync(file.fileno())
            self._sequence += 1
