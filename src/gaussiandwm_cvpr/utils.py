from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML file must contain a mapping: {path}")
    return payload


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(payload: Any, path: str | Path, *, indent: int = 2) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def load_json_or_jsonl(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        items: list[Any] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items
    payload = load_json(input_path)
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array or JSONL records: {path}")
    return payload


def dump_jsonl(items: Iterable[Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if base is None:
        return path.resolve()
    return (Path(base) / path).resolve()


def timestamp_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
