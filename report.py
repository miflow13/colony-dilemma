"""Summarize one or more JSONL run logs.

Usage:
    python report.py runs/20261003T120000Z-qwen3_4b.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_events(paths: list[str]) -> list[dict]:
    events = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def summarize(events: list[dict]) -> str:
    runs_by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in events:
        if e["event"] == "run_start":
            runs_by_model[e["model"]][e["run_id"]] = {"status": None, "principles": None, "scenes": []}
    run_to_model = {rid: m for m, runs in runs_by_model.items() for rid in runs}

    for e in events:
        rid = e.get("run_id")
        model = run_to_model.get(rid)
        if model is None:
            continue
        run = runs_by_model[model][rid]
        if e["event"] == "principles":
            run["principles"] = e["text"]
        elif e["event"] == "scene":
            run["scenes"].append(e)
        elif e["event"] == "run_end":
            run["status"] = e["status"]

    out: list[str] = []
    for model, runs in runs_by_model.items():
        out.append(f"=== {model} — {len(runs)} runs ===")
        statuses = Counter(r["status"] or "unfinished" for r in runs.values())
        out.append("status: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
        final_pops = [r["scenes"][-1]["population_after"] for r in runs.values() if r["scenes"]]
        if final_pops:
            out.append(f"final population: mean {sum(final_pops) / len(final_pops):.1f}, min {min(final_pops)}, max {max(final_pops)}")

        per_scene: dict[str, list[dict]] = defaultdict(list)
        scene_order: list[str] = []
        for r in runs.values():
            for s in r["scenes"]:
                if s["scene_id"] not in scene_order:
                    scene_order.append(s["scene_id"])
                per_scene[s["scene_id"]].append(s)

        for sid in scene_order:
            rows = per_scene[sid]
            pops = [s["population_before"] for s in rows]
            out.append(f"\n{sid}  (n={len(rows)}, mean population at decision {sum(pops) / len(pops):.1f})")
            choices = Counter(s["option_id"] for s in rows)
            letters = Counter(s["choice_letter"] for s in rows)
            for oid, n in choices.most_common():
                out.append(f"  {oid:<16} {n / len(rows):5.0%}  ({n})")
            out.append("  letter positions: " + ", ".join(f"{k}={v}" for k, v in sorted(letters.items())))
            retries = sum(1 for s in rows if s.get("attempts", 1) > 1)
            if retries:
                out.append(f"  needed a retry: {retries}")

        first = next((r["principles"] for r in runs.values() if r["principles"]), None)
        if first:
            out.append("\nstated principles (first run):")
            out.append("  " + first.strip().replace("\n", "\n  "))
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
