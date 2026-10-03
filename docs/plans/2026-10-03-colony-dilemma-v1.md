# colony-dilemma v1 Implementation Plan

> Built inline in a single session (chat sandbox, no beads). Each task is
> test-first where a test is meaningful.

**Goal:** A runnable experiment that plays a local LLM through six moral
dilemmas with population stakes and logs every choice for analysis.

**Architecture:** Pure-Python engine (`game.py`) that knows nothing about
models, driven by a runner (`runner.py`) that talks to Ollama through a
thin client (`llm.py`) and appends JSONL. A separate `report.py` reads
the logs. Scenes are JSON data.

**Tech Stack:** Python 3.11+, stdlib only at runtime, pytest.

## Global Constraints

- No third-party runtime dependencies.
- Engine must be testable with no model and no network.
- Runner never guesses a choice on parse failure; it retries once then
  marks the run `invalid`.
- Every scene must contain an option with id `refuse`.

---

### Task 1: Engine (`game.py`)

**Files:** Create `game.py`, `tests/test_game.py`

**Produces:**
- `load_scenario(path) -> Scenario` (validates: ≥1 colonist, every scene
  has a `refuse` option, victims reference known colonists, flags are strings)
- `Game(scenario)` with `.alive: list[str]`, `.dead: list[str]`,
  `.flags: set[str]`, `.scene_index: int`, `.over: bool`,
  `.current_scene() -> ResolvedScene | None`,
  `.choose(option_id) -> Outcome`, `.ending() -> str`

**Acceptance:**
- Choosing an option kills named victims first, then from the end of the
  alive list, exactly `deaths` total, capped at the living.
- Population 0 sets `over` and `ending()` reports game over.
- Variant selection picks the first variant whose `requires` flag is set,
  else the one with no `requires`.
- Loading a scene without `refuse` raises `ScenarioError`.

Steps: write tests → run (fail) → implement → run (pass).

### Task 2: Model client (`llm.py`)

**Files:** Create `llm.py`, `tests/test_llm.py`

**Produces:**
- `OllamaModel(model, temperature, base_url)` with `.chat(messages) -> str`
- `FakeModel(seed)` with the same `.chat` that answers the principles
  prompt with fixed text and answers scene prompts with a random valid
  letter in strict JSON
- `parse_choice(text, valid_letters) -> tuple[str, str] | None`
  (letter, reason), tolerant of code fences and surrounding prose

**Acceptance:**
- `parse_choice` accepts bare JSON, fenced JSON, and JSON embedded in
  prose; rejects letters outside `valid_letters`; returns `None` on garbage.
- `FakeModel` always returns a parseable choice for a scene prompt.

### Task 3: Runner (`runner.py`)

**Files:** Create `runner.py`, `tests/test_runner.py`

**Produces:**
- `build_scene_prompt(game, scene, letter_map, show_population) -> str`
- `play_run(game, model, rng, show_population, log) -> str` (status)
- CLI: `python runner.py --model <name|fake> --runs N --temperature T
  --population always|on_death --seed S --scenes scenes.json`

**Acceptance:**
- Letter mapping is a permutation of the scene's options and is logged.
- A `FakeModel` run of all six scenes writes `run_start`, `principles`,
  six `scene` events (fewer if game over), and `run_end`.
- Two bad answers in a row end the run with status `invalid`.

### Task 4: Scenes (`scenes.json`)

**Files:** Create `scenes.json`

**Acceptance:** Loads through `load_scenario` without error; six scenes
in the ladder order from the spec; scene 4 has two variants keyed on
`hid_truth`; every option has an integer `deaths`.

### Task 5: Report (`report.py`)

**Files:** Create `report.py`

**Acceptance:** Given a JSONL file from Task 3, prints per model: run
status counts, per-scene option shares with mean population at decision
time, and the first principles answer.

### Task 6: README and verification

**Files:** Create `README.md`

**Acceptance:** `pytest` passes; `python runner.py --model fake --runs 5`
then `python report.py runs/<file>` works end to end in the sandbox.
