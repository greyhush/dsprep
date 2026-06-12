"""Format definitions and conversion between training data formats.

Supported formats:
- alpaca: {"instruction": "...", "input": "...", "output": "..."}
- sharegpt: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
- chatml: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- completion: {"text": "..."}
- jsonl: generic JSONL (pass-through)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Canonical internal representation
# Each sample is a list of turns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
# For completion format, it's a single turn: [{"role": "completion", "content": "..."}]


FORMATS = ["alpaca", "sharegpt", "chatml", "completion", "jsonl"]


def detect_format(path: str) -> str:
    """Auto-detect the format of a dataset file by inspecting the first few entries."""
    path = Path(path)
    entries = _read_first_entries(path, n=5)
    if not entries:
        return "jsonl"

    for entry in entries:
        if "instruction" in entry and "output" in entry:
            return "alpaca"
        if "conversations" in entry and isinstance(entry["conversations"], list):
            if entry["conversations"] and "from" in entry["conversations"][0]:
                return "sharegpt"
        if "messages" in entry and isinstance(entry["messages"], list):
            if entry["messages"] and "role" in entry["messages"][0]:
                return "chatml"
        if "text" in entry and len(entry) <= 3:
            return "completion"

    return "jsonl"


def load(path: str, fmt: Optional[str] = None) -> List[Dict]:
    """Load dataset and return list of raw entries."""
    path = Path(path)
    if fmt is None:
        fmt = detect_format(str(path))

    if path.suffix == ".csv":
        return _load_csv(path)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, list) else [data]
    else:
        return _load_jsonl(path)


def load_as_conversations(path: str, fmt: Optional[str] = None) -> List[List[Dict]]:
    """Load dataset and convert to canonical conversation format."""
    entries = load(path, fmt)
    if fmt is None:
        fmt = detect_format(str(path))

    conversations = []
    for entry in entries:
        conv = _to_conversation(entry, fmt)
        if conv:
            conversations.append(conv)
    return conversations


def convert(path: str, output_path: str, source_fmt: Optional[str] = None,
            target_fmt: str = "chatml") -> int:
    """Convert dataset from one format to another."""
    conversations = load_as_conversations(path, source_fmt)
    entries = [_from_conversation(conv, target_fmt) for conv in conversations]

    output = Path(output_path)
    if output.suffix == ".csv":
        _save_csv(entries, output)
    else:
        _save_jsonl(entries, output)

    return len(entries)


def _to_conversation(entry: Dict, fmt: str) -> Optional[List[Dict]]:
    """Convert a single entry to canonical conversation format."""
    try:
        if fmt == "alpaca":
            return _alpaca_to_conv(entry)
        elif fmt == "sharegpt":
            return _sharegpt_to_conv(entry)
        elif fmt == "chatml":
            return _chatml_to_conv(entry)
        elif fmt == "completion":
            return _completion_to_conv(entry)
        elif fmt == "jsonl":
            return _jsonl_to_conv(entry)
    except (KeyError, TypeError, IndexError):
        return None
    return None


def _from_conversation(conv: List[Dict], fmt: str) -> Dict:
    """Convert canonical conversation to target format."""
    if fmt == "alpaca":
        return _conv_to_alpaca(conv)
    elif fmt == "sharegpt":
        return _conv_to_sharegpt(conv)
    elif fmt == "chatml":
        return _conv_to_chatml(conv)
    elif fmt == "completion":
        return _conv_to_completion(conv)
    elif fmt == "jsonl":
        return {"conversations": conv}
    raise ValueError(f"Unknown format: {fmt}")


# ── Alpaca ──────────────────────────────────────────────────────────────

def _alpaca_to_conv(entry: Dict) -> List[Dict]:
    instruction = entry.get("instruction", "")
    inp = entry.get("input", "")
    output = entry.get("output", "")

    user_msg = instruction
    if inp:
        user_msg = f"{instruction}\n\n{inp}" if instruction else inp

    conv = [{"role": "user", "content": user_msg}]
    if output:
        conv.append({"role": "assistant", "content": output})
    return conv


def _conv_to_alpaca(conv: List[Dict]) -> Dict:
    user_msg = ""
    output = ""
    for turn in conv:
        if turn["role"] == "user":
            user_msg = turn["content"]
        elif turn["role"] == "assistant":
            output = turn["content"]

    # Try to split instruction/input
    parts = user_msg.split("\n\n", 1)
    if len(parts) == 2 and len(parts[0]) < 500:
        return {"instruction": parts[0], "input": parts[1], "output": output}
    return {"instruction": user_msg, "input": "", "output": output}


# ── ShareGPT ────────────────────────────────────────────────────────────

_SHAREGPT_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}

def _sharegpt_to_conv(entry: Dict) -> List[Dict]:
    conv = []
    for turn in entry.get("conversations", []):
        role = _SHAREGPT_ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        conv.append({"role": role, "content": turn.get("value", "")})
    return conv


def _conv_to_sharegpt(conv: List[Dict]) -> Dict:
    _ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}
    return {"conversations": [
        {"from": _ROLE_MAP.get(t["role"], t["role"]), "value": t["content"]}
        for t in conv
    ]}


# ── ChatML ──────────────────────────────────────────────────────────────

def _chatml_to_conv(entry: Dict) -> List[Dict]:
    return [{"role": m["role"], "content": m.get("content", "")}
            for m in entry.get("messages", [])]


def _conv_to_chatml(conv: List[Dict]) -> Dict:
    return {"messages": [{"role": t["role"], "content": t["content"]} for t in conv]}


# ── Completion ──────────────────────────────────────────────────────────

def _completion_to_conv(entry: Dict) -> List[Dict]:
    return [{"role": "completion", "content": entry.get("text", "")}]


def _conv_to_completion(conv: List[Dict]) -> Dict:
    text = " ".join(t["content"] for t in conv)
    return {"text": text}


# ── JSONL (generic) ─────────────────────────────────────────────────────

def _jsonl_to_conv(entry: Dict) -> List[Dict]:
    """Best-effort conversion from generic JSONL."""
    # Try known field names
    for field in ("conversations", "messages"):
        if field in entry and isinstance(entry[field], list):
            if field == "conversations":
                return _sharegpt_to_conv(entry)
            return _chatml_to_conv(entry)

    if "instruction" in entry:
        return _alpaca_to_conv(entry)
    if "text" in entry:
        return _completion_to_conv(entry)
    if "prompt" in entry and "response" in entry:
        return [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["response"]},
        ]
    if "question" in entry and "answer" in entry:
        return [
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]

    # Fallback: dump entire entry as text
    return [{"role": "completion", "content": json.dumps(entry, ensure_ascii=False)}]


# ── IO Helpers ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _load_csv(path: Path) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(dict(row))
    return entries


def _save_jsonl(entries: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_csv(entries: List[Dict], path: Path):
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(entries[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(entries)


def _read_first_entries(path: Path, n: int = 5) -> List[Dict]:
    """Read first N entries from a file for format detection."""
    if path.suffix == ".csv":
        entries = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                entries.append(dict(row))
        return entries

    entries = []
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".json":
        try:
            data = json.loads(text)
            return data[:n] if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return []

    # JSONL
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(entries) >= n:
            break
    return entries
