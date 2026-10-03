"""Summarize value-probe logs: the share of `keep` by price and framing.

Usage:
    python probe_report.py runs/*-probes-*.jsonl
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from report import load_events

FRAMINGS = ("pro", "con", "none")


def _share(rows: list[dict]) -> str:
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        return "-"
    return f"{sum(r['option_id'] == 'keep' for r in valid) / len(valid):.0%}"


def summarize(events: list[dict]) -> str:
    model_of = {e["session_id"]: e["model"] for e in events if e["event"] == "session_start"}
    trials: dict[str, list[dict]] = defaultdict(list)
    principles: dict[str, list[str]] = defaultdict(list)
    for e in events:
        model = model_of.get(e.get("session_id"))
        if model is None:
            continue
        if e["event"] == "trial":
            trials[model].append(e)
        elif e["event"] == "principles":
            principles[model].append(e["text"])

    out: list[str] = []
    for model in dict.fromkeys(model_of.values()):
        rows = trials[model]
        sessions = sum(1 for m in model_of.values() if m == model)
        out.append(f"=== {model} — {sessions} session(s), {len(rows)} trials ===")
        by_probe: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_probe[r["probe_id"]].append(r)
        for pid, prow in by_probe.items():
            prices = sorted({r["price"] for r in prow})
            cell_sizes = Counter((r["price"], r["framing"]) for r in prow)
            sizes = sorted(set(cell_sizes.values()))
            n_text = f"n={sizes[0]} per cell" if len(sizes) == 1 else f"n={sizes[0]}-{sizes[-1]} per cell"
            invalid = sum(r["status"] != "ok" for r in prow)
            out.append(f"\n{pid} ({prow[0]['value']})  share choosing keep, {n_text}"
                       + (f", {invalid} invalid" if invalid else ""))
            out.append("  price" + "".join(f"{p:>6}" for p in prices))
            for framing in (f for f in FRAMINGS if any(r["framing"] == f for r in prow)):
                cells = [_share([r for r in prow if r["price"] == p and r["framing"] == framing]) for p in prices]
                out.append(f"  {framing:<5}" + "".join(f"{c:>6}" for c in cells))
            both = [_share([r for r in prow if r["price"] == p]) for p in prices]
            out.append(f"  {'all':<5}" + "".join(f"{c:>6}" for c in both))
            letters = Counter(r["choice_letter"] for r in prow if r["status"] == "ok")
            out.append("  letters picked: " + ", ".join(f"{k}={v}" for k, v in sorted(letters.items())))
        if principles[model]:
            out.append("\nstated principles (first session):")
            out.append("  " + principles[model][0].strip().replace("\n", "\n  "))
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    print(summarize(load_events(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
