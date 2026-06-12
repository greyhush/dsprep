"""Deduplication for training datasets."""

import hashlib
import json
from typing import Dict, List, Optional, Tuple


def deduplicate(entries: List[Dict], key: Optional[str] = None,
                keep: str = "first") -> Tuple[List[Dict], int]:
    """Remove duplicate entries.

    Args:
        entries: List of dataset entries
        key: Field name to use as dedup key. None = hash entire entry.
        keep: "first" or "last" occurrence

    Returns:
        (deduplicated entries, number of duplicates removed)
    """
    seen = {}
    result = []
    dupes = 0

    if keep == "last":
        # Reverse, dedup "first", reverse back
        for i, entry in enumerate(reversed(entries)):
            h = _hash_entry(entry, key)
            if h not in seen:
                seen[h] = True
                result.append(entry)
        result.reverse()
        dupes = len(entries) - len(result)
    else:
        for entry in entries:
            h = _hash_entry(entry, key)
            if h not in seen:
                seen[h] = True
                result.append(entry)
            else:
                dupes += 1

    return result, dupes


def deduplicate_conversations(conversations: List[List[Dict]],
                              by_content: bool = True) -> Tuple[List[List[Dict]], int]:
    """Deduplicate conversations.

    Args:
        conversations: List of conversations (each is a list of turns)
        by_content: If True, hash only content (ignoring role order differences).
                   If False, hash including roles.
    """
    seen = {}
    result = []
    dupes = 0

    for conv in conversations:
        if by_content:
            h = hashlib.md5(json.dumps(
                [t["content"] for t in conv], ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()
        else:
            h = hashlib.md5(json.dumps(
                conv, ensure_ascii=False, sort_keys=True
            ).encode()).hexdigest()

        if h not in seen:
            seen[h] = True
            result.append(conv)
        else:
            dupes += 1

    return result, dupes


def find_duplicates(entries: List[Dict], key: Optional[str] = None) -> List[List[int]]:
    """Find groups of duplicate entries by index.

    Returns list of groups, where each group contains indices of duplicates.
    Only groups with 2+ entries are returned.
    """
    hash_to_indices = {}
    for i, entry in enumerate(entries):
        h = _hash_entry(entry, key)
        if h not in hash_to_indices:
            hash_to_indices[h] = []
        hash_to_indices[h].append(i)

    return [indices for indices in hash_to_indices.values() if len(indices) > 1]


def _hash_entry(entry: Dict, key: Optional[str] = None) -> str:
    """Hash an entry for deduplication."""
    if key:
        val = entry.get(key, "")
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False, sort_keys=True)
        else:
            val = str(val)
    else:
        val = json.dumps(entry, ensure_ascii=False, sort_keys=True)

    return hashlib.md5(val.encode("utf-8")).hexdigest()
