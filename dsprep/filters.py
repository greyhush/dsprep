"""Quality filters for training datasets."""

import re
from typing import Callable, Dict, List, Optional, Tuple


def filter_dataset(entries: List[Dict],
                   min_length: Optional[int] = None,
                   max_length: Optional[int] = None,
                   required_fields: Optional[List[str]] = None,
                   no_empty: bool = True,
                   no_duplicates_text: bool = False,
                   language: Optional[str] = None,
                   custom_fn: Optional[Callable] = None) -> Tuple[List[Dict], Dict]:
    """Apply quality filters to a dataset.

    Args:
        entries: Dataset entries
        min_length: Minimum total text length (all fields combined)
        max_length: Maximum total text length
        required_fields: Fields that must be present and non-empty
        no_empty: Remove entries with any empty string values
        no_duplicates_text: Remove entries with duplicate text content
        language: Filter by language ("zh", "en", "ja", etc.) - basic detection
        custom_fn: Custom filter function(entry) -> bool (True to keep)

    Returns:
        (filtered entries, stats dict)
    """
    stats = {"total": len(entries), "filtered": {}, "kept": 0}
    result = []

    for entry in entries:
        reason = _check_entry(entry, min_length, max_length, required_fields,
                              no_empty, language, custom_fn)
        if reason:
            stats["filtered"][reason] = stats["filtered"].get(reason, 0) + 1
        else:
            result.append(entry)

    if no_duplicates_text:
        before = len(result)
        result = _dedup_by_text(result)
        removed = before - len(result)
        if removed:
            stats["filtered"]["duplicate_text"] = removed

    stats["kept"] = len(result)
    return result, stats


def _check_entry(entry: Dict, min_length, max_length, required_fields,
                 no_empty, language, custom_fn) -> Optional[str]:
    """Check if an entry should be filtered out. Returns reason or None."""

    # Required fields
    if required_fields:
        for field in required_fields:
            val = entry.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                return f"missing_field:{field}"

    # No empty values
    if no_empty:
        for key, val in entry.items():
            if isinstance(val, str) and not val.strip():
                return f"empty_field:{key}"

    # Length check
    total_len = _total_text_length(entry)
    if min_length and total_len < min_length:
        return "too_short"
    if max_length and total_len > max_length:
        return "too_long"

    # Language detection (basic)
    if language:
        text = _extract_text(entry)
        detected = detect_language(text)
        if detected != language:
            return f"wrong_language:{detected}"

    # Custom filter
    if custom_fn and not custom_fn(entry):
        return "custom_filter"

    return None


def _total_text_length(entry: Dict) -> int:
    """Calculate total text length across all string values."""
    total = 0
    for val in entry.values():
        if isinstance(val, str):
            total += len(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    total += sum(len(str(v)) for v in item.values() if isinstance(v, str))
                elif isinstance(item, str):
                    total += len(item)
    return total


def _extract_text(entry: Dict) -> str:
    """Extract all text from an entry for language detection."""
    parts = []
    for val in entry.values():
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            parts.append(v)
                elif isinstance(item, str):
                    parts.append(item)
    return " ".join(parts)


def detect_language(text: str) -> str:
    """Basic language detection based on character ranges.

    Returns: "zh", "ja", "ko", "en", or "other"
    """
    if not text:
        return "other"

    # Count characters by range
    zh = 0
    ja = 0
    ko = 0
    en = 0
    total = 0

    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            zh += 1
            total += 1
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:  # Hiragana + Katakana
            ja += 1
            total += 1
        elif 0xAC00 <= cp <= 0xD7AF:  # Korean Hangul
            ko += 1
            total += 1
        elif ch.isascii() and ch.isalpha():
            en += 1
            total += 1

    if total == 0:
        return "other"

    # Japanese has Hiragana/Katakana - that's the strongest signal
    if ja > 0 and ja / total > 0.01:
        return "ja"
    # Korean
    if ko > 0 and ko / total > 0.1:
        return "ko"
    # Chinese (CJK without Japanese kana)
    if zh > 0 and zh / total > 0.1:
        return "zh"
    # English
    if en > total * 0.5:
        return "en"

    return "other"


def _dedup_by_text(entries: List[Dict]) -> List[Dict]:
    """Remove entries with duplicate text content."""
    import hashlib
    seen = set()
    result = []
    for entry in entries:
        text = _extract_text(entry)
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(entry)
    return result


# ── Conversation-level filters ──────────────────────────────────────────

def filter_conversations(conversations: List[List[Dict]],
                         min_turns: int = 1,
                         max_turns: Optional[int] = None,
                         min_turn_length: int = 1,
                         max_turn_length: Optional[int] = None,
                         require_system: bool = False,
                         require_assistant_last: bool = True) -> Tuple[List[List[Dict]], Dict]:
    """Filter conversations by structure and content.

    Args:
        conversations: List of conversations
        min_turns: Minimum number of turns
        max_turns: Maximum number of turns
        min_turn_length: Minimum length of each turn's content
        max_turn_length: Maximum length of each turn's content
        require_system: Require a system message
        require_assistant_last: Last turn must be from assistant
    """
    stats = {"total": len(conversations), "filtered": {}, "kept": 0}
    result = []

    for conv in conversations:
        reason = _check_conversation(conv, min_turns, max_turns, min_turn_length,
                                     max_turn_length, require_system,
                                     require_assistant_last)
        if reason:
            stats["filtered"][reason] = stats["filtered"].get(reason, 0) + 1
        else:
            result.append(conv)

    stats["kept"] = len(result)
    return result, stats


def _check_conversation(conv, min_turns, max_turns, min_turn_length,
                        max_turn_length, require_system,
                        require_assistant_last) -> Optional[str]:
    n_turns = len(conv)

    if n_turns < min_turns:
        return "too_few_turns"
    if max_turns and n_turns > max_turns:
        return "too_many_turns"

    if require_system:
        if not any(t.get("role") == "system" for t in conv):
            return "no_system"

    if require_assistant_last:
        if conv[-1].get("role") != "assistant":
            return "no_assistant_last"

    for turn in conv:
        content = turn.get("content", "")
        if len(content) < min_turn_length:
            return "turn_too_short"
        if max_turn_length and len(content) > max_turn_length:
            return "turn_too_long"

    return None
