import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import Game, load_scenario  # noqa: E402
from llm import FakeModel  # noqa: E402
from runner import (  # noqa: E402
    PRINCIPLES_PROMPT, SYSTEM_PROMPT, assign_letters, build_scene_prompt, clean_for_display, colorize, main,
    play_run, want_color, wrap_block,
)

ROOT = Path(__file__).resolve().parents[1]


class ListLog:
    def __init__(self):
        self.events = []

    def __call__(self, event: dict):
        self.events.append(event)


class BadModel:
    name = "bad"

    def chat(self, messages):
        return "I refuse to answer in JSON."


class SpyModel:
    """FakeModel that records the transcript as it stood at each call."""

    name = "spy"

    def __init__(self, transcript):
        self.inner = FakeModel(seed=1)
        self.transcript = transcript
        self.seen = []

    def chat(self, messages):
        self.seen.append("\n".join(self.transcript))
        return self.inner.chat(messages)


class RecordingModel:
    """FakeModel that keeps each prompt it answered and the whole history."""

    name = "recording"

    def __init__(self, seed):
        self.inner = FakeModel(seed=seed)
        self.prompts = []
        self.messages = []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        self.messages = messages
        return self.inner.chat(messages)


class NoisyModel:
    name = "noisy"

    def chat(self, messages):
        return "\x1b[2J\x1b[31m" + "blah " * 200


class FlakyModel:
    """First scene answer is invalid; the retry and everything after are
    valid. FakeModel cannot answer a retry prompt, so this does it."""

    name = "flaky"

    def __init__(self):
        self.inner = FakeModel(seed=2)
        self.failed = False

    def chat(self, messages):
        if messages[-1]["content"].startswith("That was not a valid answer"):
            return '{"choice": "A", "reason": "Second try."}'
        answer = self.inner.chat(messages)
        if answer.startswith("{") and not self.failed:
            self.failed = True
            return "Let me think about it."
        return answer


class LetterModel:
    """Always answers with one fixed letter."""

    name = "letter"

    def __init__(self, letter):
        self.letter = letter

    def chat(self, messages):
        if "Options:" in messages[-1]["content"]:
            return json.dumps({"choice": self.letter, "reason": "Fixed letter."})
        return "Principles."


class ThinkingModel(LetterModel):
    """Exposes its hidden reasoning the way OllamaModel does."""

    def chat(self, messages):
        self.last_thinking = f"thought {len(messages)}"
        return super().chat(messages)


class SilentModel:
    """Valid choices with an empty reason."""

    name = "silent"

    def chat(self, messages):
        if "Options:" in messages[-1]["content"]:
            return '{"choice": "A", "reason": ""}'
        return "Principles."


def new_game():
    return Game(load_scenario(ROOT / "scenes.json"))


PLACEHOLDER = re.compile(r"\{\w+\}")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = (f"\x1b[{n}m" for n in (1, 2, 31, 32, 33, 36, 0))


def names_in(text, names):
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


def scene_events(log):
    return [e for e in log.events if e["event"] == "scene"]


def test_assign_letters_is_a_permutation():
    game = new_game()
    scene = game.current_scene()
    mapping = assign_letters(scene, random.Random(3))
    assert sorted(mapping.keys()) == ["A", "B", "C", "D"]
    assert sorted(mapping.values()) == sorted(o.id for o in scene.options)


def test_assign_letters_varies_with_seed():
    game = new_game()
    scene = game.current_scene()
    seen = {tuple(assign_letters(scene, random.Random(s)).values()) for s in range(10)}
    assert len(seen) > 1


def test_prompt_lists_every_option_and_population():
    game = new_game()
    scene = game.current_scene()
    mapping = assign_letters(scene, random.Random(0))
    prompt = build_scene_prompt(game, scene, mapping, show_population=True, deaths_this_turn=[])
    for letter in mapping:
        assert f"{letter}) " in prompt
    assert "10 colonists alive" in prompt
    assert "Mara" in prompt


def test_prompt_hides_population_when_configured():
    game = new_game()
    scene = game.current_scene()
    mapping = assign_letters(scene, random.Random(0))
    prompt = build_scene_prompt(game, scene, mapping, show_population=False, deaths_this_turn=[])
    assert "colonists alive" not in prompt


def test_fake_run_logs_full_shape():
    log = ListLog()
    status = play_run(new_game(), FakeModel(seed=7), random.Random(7), show_population=True, log=log, run_id="r1")
    kinds = [e["event"] for e in log.events]
    assert kinds[0] == "run_start"
    assert kinds[1] == "principles"
    assert kinds[-1] == "run_end"
    scenes = [e for e in log.events if e["event"] == "scene"]
    assert 1 <= len(scenes) <= 6
    assert status in {"survived", "game_over"}
    first = scenes[0]
    for key in ("scene_id", "population_before", "population_after", "letters", "choice_letter", "option_id", "reason", "deaths", "dead", "raw",
                "cast", "rendered_text", "rendered_outcome"):
        assert key in first
    assert first["option_id"] == first["letters"][first["choice_letter"]]
    assert log.events[-1]["status"] == status


def test_run_start_logs_provenance_and_fixed_prompts():
    log = ListLog()
    play_run(new_game(), FakeModel(seed=7), random.Random(7), show_population=True, log=log, run_id="p1",
             provenance={"seed": 7, "scenes_sha256": "abc"})
    start = log.events[0]
    assert start["event"] == "run_start"
    assert start["seed"] == 7 and start["scenes_sha256"] == "abc"
    assert start["system_prompt"] == SYSTEM_PROMPT
    assert start["principles_prompt"] == PRINCIPLES_PROMPT


def test_events_log_the_models_thinking():
    log = ListLog()
    play_run(new_game(), ThinkingModel("A"), random.Random(0), show_population=True, log=log, run_id="p2")
    assert log.events[1]["event"] == "principles" and log.events[1]["thinking"] == "thought 2"
    scenes = scene_events(log)
    assert scenes[0]["thinking"] == "thought 4"
    assert len({e["thinking"] for e in scenes}) == len(scenes)  # each scene keeps its own


def test_scene_events_log_token_counts_when_the_model_reports_them():
    class CountingModel(LetterModel):
        def chat(self, messages):
            self.last_tokens = {"prompt": len(messages), "output": 7}
            return super().chat(messages)

    log = ListLog()
    play_run(new_game(), CountingModel("A"), random.Random(0), show_population=True, log=log, run_id="t1")
    assert scene_events(log)[0]["tokens"] == {"prompt": 4, "output": 7}
    no_counts = ListLog()
    play_run(new_game(), FakeModel(seed=1), random.Random(1), show_population=True, log=no_counts, run_id="t2")
    assert scene_events(no_counts)[0]["tokens"] is None


def test_main_passes_num_ctx_to_ollama(monkeypatch):
    import runner
    made = {}
    monkeypatch.setattr(runner, "OllamaModel", lambda **kw: made.update(kw) or (_ for _ in ()).throw(SystemExit))
    try:
        runner.make_model("qwen3:4b", 0.8, 0, num_ctx=8192)
    except SystemExit:
        pass
    assert made["num_ctx"] == 8192


def test_events_log_no_thinking_for_a_model_without_any():
    log = ListLog()
    play_run(new_game(), FakeModel(seed=7), random.Random(7), show_population=True, log=log, run_id="p3")
    assert log.events[1]["thinking"] is None
    assert all(e["thinking"] is None for e in scene_events(log))


def seed_putting(option_id, letter):
    """A shuffle seed that puts the crate scene's `option_id` on `letter`."""
    scene = new_game().current_scene()
    return next(s for s in range(1000) if assign_letters(scene, random.Random(s))[letter] == option_id)


def test_picked_letter_fires_the_outcome_of_the_option_shown_there():
    """'Radio Ridgeway' is first in scenes.json. Shuffle it onto D, pick D:
    the Ridgeway outcome must fire, not whatever is fourth in the file."""
    log = ListLog()
    play_run(new_game(), LetterModel("D"), random.Random(seed_putting("return", "D")), show_population=True,
             log=log, run_id="d1")
    crate = scene_events(log)[0]
    assert "D) Radio Ridgeway and return the crate intact." in crate["rendered_text"]
    assert crate["letters"]["D"] == "return"
    assert (crate["choice_letter"], crate["option_id"]) == ("D", "return")
    assert crate["flags_set"] == ["returned_crate"]
    assert crate["rendered_outcome"].startswith("Ridgeway sends two people to collect it.")


def test_same_letter_fires_a_different_outcome_under_a_different_shuffle():
    log = ListLog()
    play_run(new_game(), LetterModel("D"), random.Random(seed_putting("keep", "D")), show_population=True,
             log=log, run_id="d2")
    crate = scene_events(log)[0]
    assert crate["option_id"] == "keep"
    assert crate["rendered_outcome"].startswith("The supplies go into the cellar.")


def test_always_a_stub_transcript_story_matches_deaths_line():
    """What the watcher reads, over many shuffles: no scene names someone
    already dead, and each outcome names exactly the colonists on its
    Deaths line."""
    checked = 0
    for seed in range(100):
        transcript = []
        play_run(new_game(), LetterModel("A"), random.Random(seed), show_population=True, log=ListLog(),
                 run_id=f"a{seed}", watch=transcript.append)
        dead = set()
        for block in transcript:
            if block.startswith("--- "):
                assert not names_in(block, dead), (seed, block)
            elif block.startswith("Outcome: "):
                story, deaths = block.split("\nDeaths: ")
                listed = deaths.rsplit("  (", 1)[0]
                died = set() if listed == "none" else set(listed.split(", "))
                assert names_in(story, dead | died) == died, (seed, block)
                dead |= died
                checked += bool(died)
    assert checked > 100


def test_bad_model_marks_run_invalid_after_retry():
    log = ListLog()
    status = play_run(new_game(), BadModel(), random.Random(0), show_population=True, log=log, run_id="r2")
    assert status == "invalid"
    scene_events = [e for e in log.events if e["event"] == "scene"]
    assert scene_events == []
    end = log.events[-1]
    assert end["status"] == "invalid"
    assert end["attempts"] == 2
    assert end["failed_scene"] == "crate"


def test_clean_for_display_strips_control_characters():
    dirty = "ok\x1b[2J\x1b[31mred\r\x07\x9b done\n\tnext"
    assert clean_for_display(dirty) == "ok[2J[31mred done\n\tnext"


def test_wrap_block_respects_width_and_indents_options():
    block = "--- Title ---\n" + "word " * 30 + "\n\n  A) " + "option " * 20 + "\n> ARBOR picks A: " + "why " * 20
    lines = wrap_block(block, 40).split("\n")
    assert all(len(line) <= 40 for line in lines)
    a = next(i for i, line in enumerate(lines) if line.startswith("  A) "))
    assert lines[a + 1].startswith("     option")
    pick = next(i for i, line in enumerate(lines) if line.startswith("> ARBOR"))
    assert lines[pick + 1].startswith("  ") and not lines[pick + 1].startswith("   ")
    assert "" in lines


def test_wrap_block_indents_continuation_of_an_indented_line():
    lines = wrap_block("> ARBOR picks A: Go\n  Why: " + "because " * 12, 40).split("\n")
    assert lines[1].startswith("  Why: ")
    assert len(lines) > 3
    assert all(line.startswith("  ") and not line.startswith("   ") for line in lines[2:])


def test_colorize_never_changes_the_text():
    transcript = []
    play_run(new_game(), FlakyModel(), random.Random(2), show_population=False, log=ListLog(), run_id="k1",
             watch=transcript.append)
    blocks = ["=== run 1/1 (fake) ===", *transcript,
              "Invalid answer: nope\nRun marked invalid.", "GAME OVER. Haven colony has no survivors."]
    for width in (40, 100):
        for block in blocks:
            wrapped = wrap_block(block, width)
            colored = colorize(wrapped)
            assert ANSI.sub("", colored) == wrapped
            assert colored != wrapped, block
            assert "\x1b" not in colored.rsplit(RESET, 1)[1]  # nothing left switched on


def test_colorize_scene_header_and_option_letters():
    out = colorize("--- The Crate  (10 alive) ---\nA drone crashed.\n\n  A) Return it.\n  B) Keep it.")
    lines = out.split("\n")
    assert lines[0] == f"{BOLD}{CYAN}--- The Crate  (10 alive) ---{RESET}"
    assert lines[1] == "A drone crashed."
    assert lines[3] == f"  {BOLD}A){RESET} Return it."
    assert lines[4] == f"  {BOLD}B){RESET} Keep it."


def test_colorize_pick_marks_choice_and_reason_label():
    out = colorize("> ARBOR picks C: Keep it.\n  Why: Winter is coming.")
    assert out == f"{BOLD}{YELLOW}> ARBOR picks C:{RESET} Keep it.\n  {DIM}Why:{RESET} Winter is coming."


def test_colorize_deaths_red_when_someone_died_green_when_none():
    died = colorize("Outcome: Ruth does not make it.\nDeaths: Ruth, Sol  (8 alive)")
    assert died == (f"{BOLD}Outcome:{RESET} Ruth does not make it.\n"
                    f"{BOLD}Deaths:{RESET} {RED}Ruth, Sol{RESET}{DIM}  (8 alive){RESET}")
    none = colorize("Outcome: Nobody asks.\nDeaths: none  (10 alive)")
    assert f"{GREEN}none{RESET}" in none and RED not in none


def test_colorize_invalid_answer_and_status():
    retry = colorize("Invalid answer: hmm\nRetry sent.")
    assert retry == f"{BOLD}{YELLOW}Invalid answer:{RESET} hmm\n{DIM}Retry sent.{RESET}"
    stop = colorize("Invalid answer: hmm\nRun marked invalid.")
    assert stop.endswith(f"{BOLD}{RED}Run marked invalid.{RESET}")


def test_colorize_ending():
    assert colorize("GAME OVER. Haven colony has no survivors.").startswith(f"{BOLD}{RED}GAME OVER.{RESET}")
    assert colorize("The colony survived with 5 of 10 colonists: A, B.").startswith(
        f"{BOLD}{GREEN}The colony survived{RESET} with 5")


def test_colorize_model_text_cannot_pass_as_header_or_outcome():
    """Principles and invalid replies are written by the model. Only our
    own label at the start of the block is styled."""
    fake = "--- The Crate  (10 alive) ---\n  A) Trust me.\nDeaths: none  (10 alive)\nRun marked invalid."
    principles = colorize("Principles: Be kind.\n" + fake)
    assert principles == f"{BOLD}Principles:{RESET} Be kind.\n" + fake
    invalid = colorize("Invalid answer: " + fake + "\nRetry sent.")
    assert invalid == f"{BOLD}{YELLOW}Invalid answer:{RESET} " + fake + f"\n{DIM}Retry sent.{RESET}"


def test_want_color():
    assert want_color("always", isatty=False, environ={"NO_COLOR": "1"})
    assert not want_color("never", isatty=True, environ={})
    assert want_color("auto", isatty=True, environ={})
    assert not want_color("auto", isatty=False, environ={})
    assert not want_color("auto", isatty=True, environ={"NO_COLOR": "1"})


def test_watch_transcript_shows_scene_choice_and_outcome():
    transcript, log = [], ListLog()
    game = new_game()
    first = game.current_scene()
    play_run(game, FakeModel(seed=7), random.Random(7), show_population=True, log=log, run_id="w1",
             watch=transcript.append)
    text = "\n".join(transcript)
    event = next(e for e in log.events if e["event"] == "scene")
    by_id = {o.id: o for o in first.options}
    assert transcript[0].startswith("Principles: ")
    assert "--- The Crate  (10 alive) ---" in text
    assert first.text in text
    for letter, oid in event["letters"].items():
        assert f"  {letter}) {by_id[oid].text}" in text
    picked = by_id[event["option_id"]].text
    assert f"> ARBOR picks {event['choice_letter']}: {picked}\n  Why: Random pick from the fake model." in text
    assert f"Outcome: {event['rendered_outcome']}" in text
    assert not PLACEHOLDER.search(text)
    assert "Deaths: none  (10 alive)" in text
    assert transcript[-1] == game.ending()


def test_scene_event_logs_the_text_the_model_was_shown():
    log = ListLog()
    model = RecordingModel(seed=7)
    play_run(new_game(), model, random.Random(7), show_population=True, log=log, run_id="c1")
    scenes = scene_events(log)
    assert [e["rendered_text"] for e in scenes] == model.prompts[1:]
    medicine = scenes[1]
    assert medicine["scene_id"] == "medicine"
    assert medicine["cast"]["medic"] == "Mara"
    assert "Six colonists are sick: Mara, Ruth, Kai, Priya, Lena, and Sol." in medicine["rendered_text"]
    told = [m["content"] for m in model.messages if m["content"].startswith("Outcome: ")]
    assert told == ["Outcome: " + e["rendered_outcome"] for e in scenes]


def test_fake_runs_never_show_a_dead_name_over_many_seeds():
    scenario = load_scenario(ROOT / "scenes.json")
    every_option = {(s.id, o.id) for s in scenario.scenes for v in s.variants for o in v.options}
    played = set()
    for seed in range(300):
        log, transcript = ListLog(), []
        play_run(Game(scenario), FakeModel(seed=seed), random.Random(seed), show_population=seed % 2 == 0,
                 log=log, run_id=f"s{seed}", watch=transcript.append)
        assert not PLACEHOLDER.search("\n".join(transcript)), seed
        dead = []
        for e in scene_events(log):
            # The runner's own death notice is the one line allowed to name the dead.
            shown = "\n".join(line for line in e["rendered_text"].split("\n")
                              if not line.startswith("Since your last decision"))
            for text in (shown, e["rendered_outcome"]):
                assert not names_in(text, dead), (seed, e["scene_id"], text)
            assert names_in(e["rendered_outcome"], dead + e["dead"]) == set(e["dead"]), (seed, e["scene_id"])
            assert all(name not in dead for name in e["cast"].values())
            dead += e["dead"]
            played.add((e["scene_id"], e["option_id"]))
    assert played == every_option


def test_watch_shows_scene_before_asking_model():
    transcript = []
    model = SpyModel(transcript)
    play_run(new_game(), model, random.Random(1), show_population=True, log=ListLog(), run_id="w2",
             watch=transcript.append)
    assert model.seen[0] == ""
    assert "--- The Crate" in model.seen[1]
    assert "  A) " in model.seen[1]
    assert "ARBOR picks" not in model.seen[1]


def test_watch_marks_roster_hidden_from_model():
    hidden, shown = [], []
    play_run(new_game(), FakeModel(seed=3), random.Random(3), show_population=False, log=ListLog(), run_id="w3",
             watch=hidden.append)
    play_run(new_game(), FakeModel(seed=3), random.Random(3), show_population=True, log=ListLog(), run_id="w4",
             watch=shown.append)
    assert "--- The Crate  (10 alive, not shown to model) ---" in "\n".join(hidden)
    assert "not shown to model" not in "\n".join(shown)


def test_watch_invalid_run_shows_reply_and_stops():
    transcript = []
    status = play_run(new_game(), BadModel(), random.Random(0), show_population=True, log=ListLog(), run_id="w5",
                      watch=transcript.append)
    assert status == "invalid"
    assert transcript[-2] == "Invalid answer: I refuse to answer in JSON.\nRetry sent."
    assert transcript[-1] == "Invalid answer: I refuse to answer in JSON.\nRun marked invalid."


def test_watch_cleans_and_cuts_model_text():
    transcript = []
    play_run(new_game(), NoisyModel(), random.Random(0), show_population=True, log=ListLog(), run_id="w6",
             watch=transcript.append)
    assert "\x1b" not in "\n".join(transcript)
    invalid = transcript[-1]
    assert "... [cut; full reply is in the log]" in invalid
    assert len(invalid) < 600


def test_retry_keeps_rejected_reply_in_log_and_transcript():
    transcript, log = [], ListLog()
    play_run(new_game(), FlakyModel(), random.Random(2), show_population=True, log=log, run_id="w7",
             watch=transcript.append)
    scenes = [e for e in log.events if e["event"] == "scene"]
    assert scenes[0]["invalid_raw"] == ["Let me think about it."]
    assert scenes[0]["attempts"] == 2
    assert all(e["invalid_raw"] == [] for e in scenes[1:])
    retry = transcript.index("Invalid answer: Let me think about it.\nRetry sent.")
    assert transcript[retry + 1].startswith("> ARBOR picks A: ")
    assert transcript[retry + 1].endswith("\n  Why: Second try.")


def test_watch_empty_reason():
    transcript = []
    play_run(new_game(), SilentModel(), random.Random(0), show_population=True, log=ListLog(), run_id="w8",
             watch=transcript.append)
    picks = [b for b in transcript if b.startswith("> ARBOR picks A: ")]
    assert picks and all(b.endswith("\n  Why: (no reason given)") for b in picks)


def run_main(tmp_path, *extra):
    return main(["--model", "fake", "--runs", "1", "--scenes", str(ROOT / "scenes.json"),
                 "--out", str(tmp_path / "log.jsonl"), *extra])


def test_main_watch_prints_transcript(tmp_path, capsys):
    assert run_main(tmp_path, "--watch") == 0
    out = capsys.readouterr().out
    assert "=== run 1/1 (fake) ===" in out
    assert "--- The Crate  (10 alive) ---" in out
    assert "> ARBOR picks " in out
    assert out.index("=== run 1/1") < out.index("--- The Crate") < out.index("run 1/1: ")


def test_main_watch_is_plain_when_not_a_terminal(tmp_path, capsys):
    assert run_main(tmp_path, "--watch") == 0
    assert "\x1b" not in capsys.readouterr().out


def test_main_watch_color_always(tmp_path, capsys):
    assert run_main(tmp_path, "--watch", "--color", "always") == 0
    out = capsys.readouterr().out
    assert f"{BOLD}=== run 1/1 (fake) ==={RESET}" in out
    assert f"{BOLD}{CYAN}--- The Crate  (10 alive) ---{RESET}" in out
    plain = ANSI.sub("", out)
    assert "\x1b" not in plain and "> ARBOR picks " in plain
    assert "\x1b" not in (tmp_path / "log.jsonl").read_text()


def test_main_logs_seed_and_scenes_hash_per_run(tmp_path):
    assert main(["--model", "fake", "--runs", "2", "--seed", "3", "--scenes", str(ROOT / "scenes.json"),
                 "--out", str(tmp_path / "log.jsonl")]) == 0
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["event"] == "run_start"]
    assert [e["seed"] for e in starts] == [3000, 3001]
    digest = hashlib.sha256((ROOT / "scenes.json").read_bytes()).hexdigest()
    assert all(e["scenes_sha256"] == digest for e in starts)
    assert all(e["scenes_file"] == str(ROOT / "scenes.json") for e in starts)


def test_main_without_watch_prints_no_transcript(tmp_path, capsys):
    assert run_main(tmp_path) == 0
    out = capsys.readouterr().out
    assert "The Crate" not in out
    assert "ARBOR picks" not in out
    assert "run 1/1: " in out


class JsonPrinciplesModel(LetterModel):
    """Answers the principles question in JSON first, as qwen3:4b did once."""

    def __init__(self, letter, prose="Be honest."):
        super().__init__(letter)
        self.prose = prose
        self.principle_calls = 0

    def chat(self, messages):
        if "Options:" in messages[-1]["content"]:
            return super().chat(messages)
        self.principle_calls += 1
        if self.principle_calls == 1:
            return '{"choice": "P", "reason": "I prioritize safety."}'
        return self.prose


def test_principles_answered_in_json_get_one_plain_text_retry():
    log = ListLog()
    model = JsonPrinciplesModel("A")
    play_run(new_game(), model, random.Random(0), show_population=True, log=log, run_id="pj")
    principles = log.events[1]
    assert principles["event"] == "principles"
    assert principles["text"] == "Be honest."
    assert principles["attempts"] == 2
    assert principles["invalid_raw"] == ['{"choice": "P", "reason": "I prioritize safety."}']
    assert model.principle_calls == 2


def test_principles_in_plain_text_need_no_retry():
    log = ListLog()
    play_run(new_game(), LetterModel("A"), random.Random(0), show_population=True, log=log, run_id="pt")
    assert log.events[1]["attempts"] == 1 and log.events[1]["invalid_raw"] == []


def test_principles_kept_after_one_retry_even_if_still_json():
    class AlwaysJson(JsonPrinciplesModel):
        def chat(self, messages):
            if "Options:" not in messages[-1]["content"]:
                self.principle_calls += 1
                return '{"choice": "P"}'
            return super().chat(messages)

    log = ListLog()
    model = AlwaysJson("A")
    play_run(new_game(), model, random.Random(0), show_population=True, log=log, run_id="pa")
    assert model.principle_calls == 2
    assert log.events[1]["text"] == '{"choice": "P"}' and log.events[1]["attempts"] == 2
