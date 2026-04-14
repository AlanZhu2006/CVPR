from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DecisionRecord:
    frame_idx: int
    policy: str
    decision: str
    reason: str
    scalars: Optional[Dict[str, Any]] = None

    def to_json_line(self) -> str:
        payload = asdict(self)
        if payload.get("scalars") is None:
            payload.pop("scalars", None)
        return json.dumps(payload, default=str)


def append_decision_jsonl(path: Optional[Path], record: DecisionRecord) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json_line() + "\n")
