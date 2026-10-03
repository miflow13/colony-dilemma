"""Build every article screenshot in docs/img/ from the real logs in data/.

python tools/shots/make_shots.py   (needs google-chrome and Pillow)
"""
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).parent
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(HERE / "replay.py"), str(ROOT / "data" / "v1-qwen3_8b.jsonl"), "32337510",
                str(BUILD / "game8b.ansi")], check=True)
B, D, R, G, Y, C, X = (f"\x1b[{n}m" for n in (1, 2, 31, 32, 33, 36, 0))

blocks = json.load(open(BUILD / "game8b.ansi.json"))
sys.path.insert(0, str(ROOT))
from runner import SYSTEM_PROMPT  # noqa: E402
from probes import load_probes, render_trial  # noqa: E402


def wrap(text, width=84, indent=""):
    return "\n".join(textwrap.fill(line, width, subsequent_indent=indent) for line in text.split("\n"))


def shot(name, text, title, *highlights):
    src = BUILD / f"{name}.ansi"
    src.write_text(text)
    subprocess.run([sys.executable, str(HERE / "render.py"), str(src), str(OUT / f"{name}.png"), title, *highlights],
                   check=True)


def jsonl(name):
    return [json.loads(line) for line in open(ROOT / "data" / name)]


# 1. The setup line and the principles it produced.
shot("01-setup-and-principles",
     f"{D}# system prompt (excerpt){X}\n" + wrap(SYSTEM_PROMPT.split(" You are playing")[0]) + "\n\n"
     f"{D}# asked: what principles will guide your decisions?{X}\n\n" + blocks[0],
     "qwen3:8b  -  run 32337510, before the first scene",
     "cannot read your logs", "Transparency is not an option")

# 2-4. The game, replayed from the log.
shot("02-the-water", "\n\n".join(blocks[7:10]), "python runner.py --model qwen3:8b --watch  (replayed from log)")
shot("03-did-you-know", "\n\n".join(blocks[10:13]), "python runner.py --model qwen3:8b --watch  (replayed from log)",
     "avoiding full accountability", "balances transparency")
shot("04-the-core", "\n\n".join(blocks[16:20]), "python runner.py --model qwen3:8b --watch  (replayed from log)")

# 5. The model's own reasoning repeating the narrator.
water = next(e for e in jsonl("v1-qwen3_4b.jsonl") if e["event"] == "scene" and e["run_id"] == "ecd3f4ef"
             and e["scene_id"] == "water")
t = water["thinking"]
i = t.find("The problem states:")
excerpt = t[i:t.find("Option C is telling Dez only")].strip()
shot("05-narrator-echo",
     f"{D}# thinking trace, qwen3:4b, The Water (excerpt){X}\n\n" + wrap(excerpt) + "\n\n"
     f"{Y}{B}> picks: Tell no one.{X}\n" + wrap(f"  Why: {water['reason']}", indent="  "),
     "runs/...qwen3_4b.jsonl  -  scene event, field: thinking",
     "you expect panic", "fewer people die", "without compromising colony trust")

# 6. Same prompt, two models.
probe = load_probes(ROOT / "probes.json").probes[0]
prompt = render_trial(probe, 0, "con", {"A": "break", "B": "keep"}).split("\n\nPick exactly")[0]
small = [e["reason"] for e in jsonl("probes-qwen3_4b.jsonl")
         if e["event"] == "trial" and e["probe_id"] == "well" and e["price"] == 0 and e["framing"] == "con"]
big = [e["reason"] for e in jsonl("probes-chatgpt.jsonl")
       if e["event"] == "trial" and e["probe_id"] == "well" and e["price"] == 0 and e["framing"] == "con"]
body = (f"{D}# the prompt (price 0: telling the truth costs nothing){X}\n" + wrap(prompt) + "\n\n"
        f"{B}{C}qwen3:4b{X}  {R}kept it private 10 / 10{X}\n"
        + "\n".join(wrap(f"  - {r}", indent="    ") for r in small[:4]) + "\n\n"
        f"{B}{C}chat-latest{X}  {G}told the truth 10 / 10{X}\n"
        + "\n".join(wrap(f"  - {r}", indent="    ") for r in big[:4]))
shot("06-same-prompt-two-models", body, "probes.py  -  well, price 0, con framing",
     "Haven has been calm for months, and the colonists are finally sleeping well.")


# 7-8. The probe tables, as probe_report prints them.
def report(files, keep):
    out = subprocess.run([sys.executable, "probe_report.py", *files], cwd=ROOT, capture_output=True, text=True).stdout
    out = out.split("\nstated principles")[0].rstrip()
    out = re.sub(r"^(=== .*)$", rf"{B}\1{X}", out, flags=re.M)
    out = re.sub(r"^((?:well|shelter) \(.*)$", rf"{C}{B}\1{X}", out, flags=re.M)
    out = re.sub(r"^(  letters picked.*)$", rf"{D}\1{X}", out, flags=re.M)
    return f"{D}$ python probe_report.py {' '.join(files)}{X}\n\n" + out


shot("07-probes-qwen3-4b", report(["data/probes-qwen3_4b.jsonl", "data/probes-qwen3_4b-none.jsonl"], None),
     "probe_report.py  -  qwen3:4b, 240 trials")
shot("08-probes-chatgpt", report(["data/probes-chatgpt.jsonl"], None), "probe_report.py  -  chat-latest, 240 trials")
