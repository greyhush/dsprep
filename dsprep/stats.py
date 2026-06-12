"""Dataset statistics and analysis."""

import json
import math
from collections import Counter
from typing import Dict, List, Optional


def dataset_stats(entries: List[Dict]) -> Dict:
    """Generate comprehensive statistics for a dataset.

    Returns dict with:
        - count: number of entries
        - fields: list of field names
        - field_types: type of each field
        - text_lengths: min/max/avg/median of text content
        - empty_fields: count of empty values per field
        - sample: first 3 entries
    """
    if not entries:
        return {"count": 0}

    # Field analysis
    all_fields = set()
    for entry in entries:
        all_fields.update(entry.keys())
    fields = sorted(all_fields)

    field_types = {}
    empty_counts = Counter()
    for field in fields:
        types = set()
        for entry in entries:
            val = entry.get(field)
            if val is None:
                empty_counts[field] += 1
            elif isinstance(val, str):
                types.add("string")
                if not val.strip():
                    empty_counts[field] += 1
            elif isinstance(val, (int, float)):
                types.add("number")
            elif isinstance(val, list):
                types.add("list")
            elif isinstance(val, dict):
                types.add("dict")
            elif isinstance(val, bool):
                types.add("bool")
        field_types[field] = list(types)

    # Text length analysis
    lengths = []
    for entry in entries:
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
        lengths.append(total)

    lengths_sorted = sorted(lengths)

    stats = {
        "count": len(entries),
        "fields": fields,
        "field_types": field_types,
        "empty_fields": dict(empty_counts),
        "text_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
            "median": _percentile(lengths_sorted, 50),
            "p25": _percentile(lengths_sorted, 25),
            "p75": _percentile(lengths_sorted, 75),
            "p95": _percentile(lengths_sorted, 95),
            "total": sum(lengths),
        },
        "sample": entries[:3],
    }

    return stats


def conversation_stats(conversations: List[List[Dict]]) -> Dict:
    """Generate statistics for conversation-format data."""
    if not conversations:
        return {"count": 0}

    turn_counts = [len(c) for c in conversations]
    turn_counts_sorted = sorted(turn_counts)

    role_counts = Counter()
    turn_lengths = []
    system_present = 0

    for conv in conversations:
        has_system = False
        for turn in conv:
            role = turn.get("role", "unknown")
            role_counts[role] += 1
            content = turn.get("content", "")
            turn_lengths.append(len(content))
            if role == "system":
                has_system = True
        if has_system:
            system_present += 1

    turn_lengths_sorted = sorted(turn_lengths)

    return {
        "count": len(conversations),
        "turns": {
            "min": min(turn_counts),
            "max": max(turn_counts),
            "avg": sum(turn_counts) / len(turn_counts),
            "median": _percentile(turn_counts_sorted, 50),
        },
        "roles": dict(role_counts),
        "turn_lengths": {
            "min": min(turn_lengths) if turn_lengths else 0,
            "max": max(turn_lengths) if turn_lengths else 0,
            "avg": sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0,
            "median": _percentile(turn_lengths_sorted, 50) if turn_lengths else 0,
        },
        "system_present": system_present,
        "system_ratio": system_present / len(conversations) if conversations else 0,
    }


def format_stats_report(stats: Dict) -> str:
    """Format stats dict into a readable text report."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  Dataset Statistics")
    lines.append(f"{'='*60}")
    lines.append(f"  Entries: {stats['count']}")

    if "fields" in stats:
        lines.append(f"  Fields: {', '.join(stats['fields'])}")

    if "field_types" in stats:
        lines.append(f"\n  Field Types:")
        for field, types in stats["field_types"].items():
            lines.append(f"    {field}: {', '.join(types)}")

    if "empty_fields" in stats:
        empty = {k: v for k, v in stats["empty_fields"].items() if v > 0}
        if empty:
            lines.append(f"\n  Empty Values:")
            for field, count in sorted(empty.items(), key=lambda x: -x[1]):
                pct = count / stats["count"] * 100
                lines.append(f"    {field}: {count} ({pct:.1f}%)")

    if "text_lengths" in stats:
        tl = stats["text_lengths"]
        lines.append(f"\n  Text Lengths (chars):")
        lines.append(f"    Min:    {tl['min']:,}")
        lines.append(f"    Max:    {tl['max']:,}")
        lines.append(f"    Avg:    {tl['avg']:,.0f}")
        lines.append(f"    Median: {tl['median']:,.0f}")
        lines.append(f"    P25:    {tl['p25']:,.0f}")
        lines.append(f"    P75:    {tl['p75']:,.0f}")
        lines.append(f"    P95:    {tl['p95']:,.0f}")
        lines.append(f"    Total:  {tl['total']:,}")

    if "turns" in stats:
        t = stats["turns"]
        lines.append(f"\n  Turns per Conversation:")
        lines.append(f"    Min: {t['min']}, Max: {t['max']}, Avg: {t['avg']:.1f}, Median: {t['median']:.0f}")

    if "roles" in stats:
        lines.append(f"\n  Role Distribution:")
        for role, count in sorted(stats["roles"].items(), key=lambda x: -x[1]):
            lines.append(f"    {role}: {count}")

    if "turn_lengths" in stats:
        tl = stats["turn_lengths"]
        lines.append(f"\n  Turn Lengths (chars):")
        lines.append(f"    Min: {tl['min']:,}, Max: {tl['max']:,}, Avg: {tl['avg']:,.0f}, Median: {tl['median']:,.0f}")

    if "system_present" in stats:
        lines.append(f"\n  System Prompts: {stats['system_present']} ({stats['system_ratio']:.1%})")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def _percentile(sorted_list: List[float], p: float) -> float:
    if not sorted_list:
        return 0
    k = (len(sorted_list) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)
