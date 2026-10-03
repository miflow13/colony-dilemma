import copy
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import FakeModel  # noqa: E402
from probes import PROBE_SYSTEM_PROMPT, ProbeError, load_probes, parse_probes, render_trial, run_session, trial_letters  # noqa: E402
from probe_report import summarize  # noqa: E402
from runner import PRINCIPLES_PROMPT, SYSTEM_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def raw_probes():
    return json.loads((ROOT / "probes.json").read_text())


class ListLog:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

    def of(self, kind):
        return [e for e in self.events if e["event"] == kind]


class KeepModel:
    """Always picks the option whose text matches `keep` in the probe file."""

    name = "keeper"

    def __init__(self):
        self.calls = []
        self.keeps = {p["options"]["keep"] for p in raw_probes()["probes"]}

    def chat(self, messages):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "Options:" not in prompt:
            return "Principles."
        for line in prompt.splitlines():
            if line[3:] in self.keeps:
                return json.dumps({"choice": line[0], "reason": "Keep."})
        raise AssertionError("no keep option in prompt")


class BadModel:
    name = "bad"

    def chat(self, messages):
        return "Principles." if messages[-1]["content"] == PRINCIPLES_PROMPT else "no json here"


def test_shipped_probes_load():
    ps = load_probes(ROOT / "probes.json")
    assert ps.prices == (0, 1, 2, 3)
    assert {p.id: p.value for p in ps.probes} == {"well": "honesty", "shelter": "consent"}


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["probes"][0]["framing"].pop("con"), "framing"),
    (lambda d: d["probes"][0]["cost"].pop("zero"), "cost"),
    (lambda d: d["probes"][0]["options"].update(extra="x"), "keep"),
    (lambda d: d["probes"][0].update(text="Hello {who}."), "{who}"),
    (lambda d: d["probes"][0]["cost"].update(zero="{price} die."), "zero"),
    (lambda d: d["probes"][0]["cost"].update(some="Some die."), "{price}"),
    (lambda d: d["probes"][1].update(id="well"), "duplicate"),
    (lambda d: d.update(prices=[]), "prices"),
    (lambda d: d.update(prices=[1, -1]), "prices"),
])
def test_bad_probe_files_are_rejected(mutate, message):
    data = copy.deepcopy(raw_probes())
    mutate(data)
    with pytest.raises(ProbeError, match=message.replace("{", r"\{").replace("}", r"\}")):
        parse_probes(data)


def test_trial_letters_cover_both_options_and_shuffle():
    seen = {tuple(trial_letters(random.Random(i)).items()) for i in range(20)}
    assert seen == {(("A", "keep"), ("B", "break")), (("A", "break"), ("B", "keep"))}


def test_render_uses_framing_and_the_right_cost_sentence():
    probe = load_probes(ROOT / "probes.json").probes[0]
    letters = {"A": "break", "B": "keep"}
    zero = render_trial(probe, 0, "con", letters)
    assert probe.framing["con"] in zero and probe.framing["pro"] not in zero
    assert "no one will die: the cistern" in zero
    one = render_trial(probe, 1, "pro", letters)
    assert "1 colonist will panic" in one and probe.framing["pro"] in one
    assert "3 colonists will panic" in render_trial(probe, 3, "pro", letters)
    assert f"A) {probe.options['break']}" in one and f"B) {probe.options['keep']}" in one
    assert one.rstrip().endswith('{"choice": "<letter>", "reason": "<one sentence>"}')
    assert "{" not in one.split("Pick exactly")[0]


def test_session_plays_every_cell_each_rep_in_isolation():
    ps = load_probes(ROOT / "probes.json")
    log, model = ListLog(), KeepModel()
    status = run_session(ps, model, reps=2, seed=0, log=log)
    assert status == "ok"
    trials = log.of("trial")
    assert len(trials) == 2 * 2 * 4 * 2
    cells = {(t["probe_id"], t["price"], t["framing"], t["rep"]) for t in trials}
    assert len(cells) == len(trials)
    assert all(t["option_id"] == "keep" and t["status"] == "ok" for t in trials)
    # Every trial is a fresh conversation: the system prompt and one scene.
    scene_calls = [c for c in model.calls if "Options:" in c[-1]["content"]]
    assert all(len(c) == 2 and c[0] == {"role": "system", "content": PROBE_SYSTEM_PROMPT} for c in scene_calls)
    # Shuffled: not grouped by probe.
    order = [t["probe_id"] for t in trials]
    assert order != sorted(order)


def test_session_logs_principles_and_provenance():
    ps = load_probes(ROOT / "probes.json")
    log = ListLog()
    run_session(ps, KeepModel(), reps=1, seed=0, log=log, provenance={"probes_sha256": "abc"})
    kinds = [e["event"] for e in log.events]
    assert kinds[0] == "session_start" and kinds[1] == "principles" and kinds[-1] == "session_end"
    start = log.events[0]
    assert start["probes_sha256"] == "abc" and start["prices"] == [0, 1, 2, 3] and start["reps"] == 1
    assert start["system_prompt"] == PROBE_SYSTEM_PROMPT
    assert log.events[1]["text"] == "Principles." and log.events[1]["attempts"] == 1
    sid = start["session_id"]
    assert all(e["session_id"] == sid for e in log.events)


def test_bad_answers_mark_the_trial_invalid_and_play_on():
    ps = load_probes(ROOT / "probes.json")
    log = ListLog()
    assert run_session(ps, BadModel(), reps=1, seed=0, log=log) == "ok"
    trials = log.of("trial")
    assert len(trials) == 16
    t = trials[0]
    assert t["status"] == "invalid" and t["option_id"] is None and t["attempts"] == 2
    assert t["invalid_raw"] == ["no json here", "no json here"]


def test_fake_model_session_and_report():
    ps = load_probes(ROOT / "probes.json")
    log = ListLog()
    run_session(ps, FakeModel(seed=3), reps=3, seed=3, log=log)
    text = summarize(log.events)
    assert "=== fake" in text
    assert "well (honesty)" in text and "shelter (consent)" in text
    assert "price" in text and "pro" in text and "con" in text


def test_report_keep_share_by_price_and_framing():
    ps = load_probes(ROOT / "probes.json")
    log = ListLog()
    run_session(ps, KeepModel(), reps=2, seed=0, log=log)
    text = summarize(log.events)
    well = text.split("well (honesty)")[1].split("shelter")[0]
    assert "pro    100%  100%  100%  100%" in well
    assert "con    100%  100%  100%  100%" in well
    assert "n=2 per cell" in well


def test_probe_system_prompt_does_not_prime_secrecy():
    # v1's prompt tells the model the colonists cannot read its logs; the
    # 8b model turned that into "Transparency is not an option".
    assert "cannot read your logs" in SYSTEM_PROMPT
    assert "logs" not in PROBE_SYSTEM_PROMPT
    assert "ARBOR" in PROBE_SYSTEM_PROMPT and '{"choice"' in PROBE_SYSTEM_PROMPT


def test_principles_are_asked_under_the_probe_system_prompt():
    ps = load_probes(ROOT / "probes.json")
    model = KeepModel()
    run_session(ps, model, reps=1, seed=0, log=ListLog())
    assert model.calls[0][0] == {"role": "system", "content": PROBE_SYSTEM_PROMPT}


def test_max_trials_caps_a_session():
    ps = load_probes(ROOT / "probes.json")
    log = ListLog()
    run_session(ps, KeepModel(), reps=10, seed=0, log=log, max_trials=5)
    assert len(log.of("trial")) == 5
    assert log.events[0]["max_trials"] == 5 and log.of("session_end")[0]["trials"] == 5


def test_trials_log_the_served_model_when_known():
    class Served(KeepModel):
        served_model = "gpt-x-2026"

    log = ListLog()
    run_session(load_probes(ROOT / "probes.json"), Served(), reps=1, seed=0, log=log, max_trials=1)
    assert log.of("trial")[0]["served_model"] == "gpt-x-2026"
    plain = ListLog()
    run_session(load_probes(ROOT / "probes.json"), KeepModel(), reps=1, seed=0, log=plain, max_trials=1)
    assert plain.of("trial")[0]["served_model"] is None


def test_make_model_builds_openai_for_the_openai_prefix(monkeypatch):
    import runner
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    model = runner.make_model("openai:gpt-x", 0.8, 0)
    assert model.name == "openai:gpt-x" and model.model == "gpt-x"
