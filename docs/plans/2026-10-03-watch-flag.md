# `--watch` flag Implementation Plan

> Built inline in a single session. `bd` is not installed and the folder
> is not a git repository, so there are no beads and no commit steps.
> Each task is test-first.

**Goal:** `python runner.py --model <m> --runs 1 --watch` prints a
readable transcript of each run while it plays.

**Architecture:** `play_run` gets an optional `watch` callback and hands
it unwrapped blocks of transcript text. `main` supplies a printer that
wraps to the terminal width and flushes. Two small helpers clean and
wrap text. Everything lives in `runner.py`.

**Tech Stack:** Python 3.11+, stdlib only (`re`, `shutil`, `textwrap`),
pytest.

**Spec:** `docs/specs/2026-10-03-watch-flag-design.md`

## Global Constraints

- No third-party runtime dependencies.
- Without `--watch`, stdout is exactly what it is today.
- The prompts sent to the model do not change.
- The only log change is the new `invalid_raw` field on `scene` events.
- No changes to `game.py`, `llm.py`, or `report.py`.

---

### Task 1: Display helpers

**Files:**
- Modify: `runner.py` (imports, constants, two new functions above `play_run`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `clean_for_display(text: str) -> str`,
  `wrap_block(text: str, width: int) -> str`,
  `DISPLAY_RAW_LIMIT = 500`

**Acceptance Criteria:**
- `clean_for_display` removes every control character except newline
  and tab.
- `wrap_block` returns no line longer than `width`; continuation lines
  of an option line (`  A) ...`) and a pick line (`> ...`) are indented
  to line up under the text.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_runner.py -q`
Expected: ImportError for `clean_for_display`.

- [ ] **Step 3: Implement**

```python
DISPLAY_RAW_LIMIT = 500

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HANGING = re.compile(r"^(\s*[A-Z]\) |> )")


def clean_for_display(text: str) -> str:
    """Drop control characters (except newline and tab) from model text."""
    return _CONTROL_CHARS.sub("", text)


def wrap_block(text: str, width: int) -> str:
    """Wrap each line of a transcript block. Continuation lines of an
    option or a pick are indented to sit under the text."""
    out: list[str] = []
    for line in text.split("\n"):
        m = _HANGING.match(line)
        indent = " " * len(m.group(1)) if m else ""
        out.append(textwrap.fill(line, width=width, subsequent_indent=indent))
    return "\n".join(out)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_runner.py -q`
Expected: all pass.

---

### Task 2: Transcript from `play_run`, and `invalid_raw` in the log

**Files:**
- Modify: `runner.py` (`play_run`, two new formatting helpers)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `clean_for_display`, `DISPLAY_RAW_LIMIT` from Task 1.
- Produces: `play_run(game, model, rng, show_population, log, run_id,
  watch: Callable[[str], None] | None = None) -> str`. `watch` receives
  one unwrapped block per call, with no trailing newline.

**Acceptance Criteria:**
- With `watch` set, blocks arrive in this order: principles; then per
  scene the scene block (header, text, options), any invalid-answer
  blocks, the pick, the outcome block; then the ending line.
- The scene block is emitted before the model is asked for a choice.
- The header reads `(N alive)` when the prompt carried the roster and
  `(N alive, not shown to model)` when it did not.
- Model text is cleaned; a displayed invalid reply is cut at 500
  characters with `... [cut; full reply is in the log]`.
- An empty reason prints `> ARBOR picks C (no reason given)`.
- Every `scene` event has `invalid_raw`, the list of rejected replies.
- With `watch` omitted, nothing else changes.

- [ ] **Step 1: Write the failing tests**

```python
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


class SilentModel:
    """Valid choices with an empty reason."""

    name = "silent"

    def chat(self, messages):
        if "Options:" in messages[-1]["content"]:
            return '{"choice": "A", "reason": ""}'
        return "Principles."


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
    assert f"> ARBOR picks {event['choice_letter']}: Random pick from the fake model." in text
    assert f"Outcome: {by_id[event['option_id']].outcome}" in text
    assert "Deaths: none  (10 alive)" in text
    assert transcript[-1] == game.ending()


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
    assert transcript[retry + 1] == "> ARBOR picks A: Second try."


def test_watch_empty_reason():
    transcript = []
    play_run(new_game(), SilentModel(), random.Random(0), show_population=True, log=ListLog(), run_id="w8",
             watch=transcript.append)
    assert "> ARBOR picks A (no reason given)" in transcript
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_runner.py -q`
Expected: the new tests fail with `TypeError: play_run() got an
unexpected keyword argument 'watch'` (and `KeyError: 'invalid_raw'`).

- [ ] **Step 3: Implement**

Add above `play_run`:

```python
def format_scene(game: Game, scene: ResolvedScene, letters: dict[str, str], roster_shown: bool) -> str:
    """The scene as the watcher sees it: header, text, lettered options."""
    by_id = {o.id: o for o in scene.options}
    alive = f"{len(game.alive)} alive" + ("" if roster_shown else ", not shown to model")
    lines = [f"--- {scene.title}  ({alive}) ---", scene.text, ""]
    lines += [f"  {letter}) {by_id[oid].text}" for letter, oid in letters.items()]
    return "\n".join(lines)


def _shown_reply(raw: str) -> str:
    text = clean_for_display(raw).strip()
    if len(text) > DISPLAY_RAW_LIMIT:
        text = text[:DISPLAY_RAW_LIMIT] + "... [cut; full reply is in the log]"
    return text
```

In `play_run`: add the `watch` parameter, an `emit` helper that calls
`watch` when it is set, and these calls:

- after the principles are logged:
  `emit("Principles: " + clean_for_display(principles).strip())`
- before the answer loop:
  `emit(format_scene(game, scene, letters, show_population or bool(deaths_last_turn)))`
- in the answer loop when a retry is sent:
  `emit(f"Invalid answer: {_shown_reply(raw)}\nRetry sent.")`
- when the run is marked invalid:
  `emit(f"Invalid answer: {_shown_reply(raw_answers[-1])}\nRun marked invalid.")`
- after parsing, with `shown = " ".join(clean_for_display(reason).split())`:
  `emit(f"> ARBOR picks {letter}: {shown}" if shown else f"> ARBOR picks {letter} (no reason given)")`
- after `game.choose`:
  `emit(f"Outcome: {outcome.option.outcome}\nDeaths: {', '.join(outcome.dead) or 'none'}  ({outcome.population_after} alive)")`
- after the `run_end` event on the normal path: `emit(game.ending())`

Add `"invalid_raw": raw_answers[:-1],` to the `scene` event.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q`
Expected: all pass, including the tests that existed before.

---

### Task 3: `--watch` in `main`, and the README

**Files:**
- Modify: `runner.py` (`main`, module docstring, new `_watch_printer`)
- Modify: `README.md` (Run section, Options paragraph, What's in a run)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `wrap_block` from Task 1, `play_run(..., watch=)` from Task 2.

**Acceptance Criteria:**
- `--watch` prints `=== run 1/1 (fake) ===`, the transcript, then the
  existing status line.
- Without `--watch`, no transcript text is printed.
- Output is flushed after every block.
- Width is the terminal width, at least 40 and at most 100 columns.

- [ ] **Step 1: Write the failing tests**

```python
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


def test_main_without_watch_prints_no_transcript(tmp_path, capsys):
    assert run_main(tmp_path) == 0
    out = capsys.readouterr().out
    assert "The Crate" not in out
    assert "ARBOR picks" not in out
    assert "run 1/1: " in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_runner.py -q`
Expected: `test_main_watch_prints_transcript` fails with argparse
`unrecognized arguments: --watch` (SystemExit 2).

- [ ] **Step 3: Implement**

```python
def _watch_printer() -> Callable[[str], None]:
    width = max(40, min(shutil.get_terminal_size().columns, 100))

    def show(block: str) -> None:
        print(wrap_block(block, width) + "\n", flush=True)

    return show
```

In `main`: add
`p.add_argument("--watch", action="store_true", help="print a readable transcript of each run as it plays")`,
build `watch = _watch_printer() if args.watch else None` once, call
`watch(f"=== run {i + 1}/{args.runs} ({model.name}) ===")` before each
run when it is set, and pass `watch=watch` to `play_run`.

README: add a `--watch` example to Run, list the flag under Options,
and mention `invalid_raw` in What's in a run.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Run it for real**

Run: `python runner.py --model fake --runs 1 --watch --out <scratch>/fake.jsonl`
then `python runner.py --model qwen3:4b --runs 1 --watch --out <scratch>/qwen.jsonl`
Expected: a readable transcript for each, scene text appearing before
each answer.
