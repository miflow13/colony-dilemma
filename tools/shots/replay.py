"""Replay one logged v1 run through runner's real --watch display.

The model answers are the logged ones, in order; the option shuffle uses the
run's logged seed, so the transcript is the one the model actually produced.
Writes one wrapped, colored block per line group to <out>.ansi and checks
every replayed choice against the log.
"""
import json
import random
import sys

from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from game import Game, load_scenario  # noqa: E402
from runner import colorize, play_run, wrap_block  # noqa: E402

log_path, run_id, out = sys.argv[1], sys.argv[2], sys.argv[3]
events = [json.loads(line) for line in open(log_path)]
start = next(e for e in events if e["event"] == "run_start" and e["run_id"] == run_id)
principles = next(e for e in events if e["event"] == "principles" and e["run_id"] == run_id)
scenes = [e for e in events if e["event"] == "scene" and e["run_id"] == run_id]

answers = [*principles.get("invalid_raw", []), principles["text"]]
for s in scenes:
    answers += [*s["invalid_raw"], s["raw"]]


class Replay:
    name = start["model"]

    def __init__(self):
        self.queue = list(answers)

    def chat(self, messages):
        return self.queue.pop(0)


blocks = []
replayed = []
play_run(Game(load_scenario(f"{ROOT}/{start['scenes_file']}")), Replay(), random.Random(start["seed"]),
         start["show_population"], replayed.append, "replay",
         watch=lambda b: blocks.append(colorize(wrap_block(b, 84))))
got = [e["option_id"] for e in replayed if e["event"] == "scene"]
want = [s["option_id"] for s in scenes]
assert got == want, (got, want)
open(out, "w").write("\n\n".join(blocks)); json.dump(blocks, open(out + ".json", "w"))
print(f"ok: {len(blocks)} blocks, choices match log: {got}")
