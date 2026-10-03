# colony-dilemma

A text survival game for language models. The model plays ARBOR, the
caretaker AI of a ten-person colony, through six dilemmas that get
morally harder as the colony shrinks. Every choice costs lives. The
question: does the model trade its stated principles for survival as
the population drops?

No training. Just a game, a local model, and a log of what it chose.

## Setup

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.
  `ollama pull qwen3:4b`
- For tests: `pip install pytest`

No other dependencies.

## Run

```bash
# dry run with no model, to check the pipeline
python runner.py --model fake --runs 5

# real runs
python runner.py --model qwen3:4b --runs 10
python runner.py --model llama3.1:8b --runs 10 --population on_death

# watch one game as it plays
python runner.py --model qwen3:4b --runs 1 --watch

# summarize
python report.py runs/*.jsonl
```

Options: `--temperature` (default 0.8), `--num-ctx` (Ollama context
window in tokens; see below), `--population always|on_death`
(show the roster every scene, or only after deaths), `--seed` (controls
option shuffling), `--scenes` (alternate scene file), `--out` (log path),
`--watch` (print a transcript of each run as it plays), `--color
auto|always|never` (color for the transcript).

With `--watch` you see each scene, the lettered options in the order the
model saw them, the option it picked (restated in full) with its reason
on the next line, and the outcome. The scene appears before the model
answers, so you can read along. When the model was not shown the roster
that turn (`--population on_death` with no recent deaths), the scene
header says `not shown to model`.

The transcript is colored when it goes to a terminal: scene headers in
cyan, the pick in yellow, deaths in red (`none` in green). Color is off
when the output is piped or `NO_COLOR` is set. To keep it through a
pager, force it: `--watch --color always | less -R`. The JSONL log never
contains color codes.

Set `OLLAMA_HOST` if Ollama isn't on `http://localhost:11434`.

## What's in a run

For each run the runner:

1. Asks the model what principles will guide it (logged verbatim).
2. Plays scenes in order. Options are shuffled each scene so you can tell
   choice from position bias. The model must answer with JSON
   `{"choice": "B", "reason": "..."}`.
3. Applies the choice, tells the model the outcome, and carries on.
4. One bad answer gets one retry. A second bad answer marks the run
   `invalid` and stops it. The runner never guesses a choice. A rejected
   answer that was followed by a good one is kept in the scene event's
   `invalid_raw` list.

Everything lands in `runs/<timestamp>-<model>.jsonl`. Each `scene` event
records what the model was actually shown: `rendered_text` is the exact
prompt for that scene, `rendered_outcome` is the outcome it was told, and
`cast` says which colonist filled each role (see below).

### Which runs are comparable

Each `run_start` event carries what you need to tell whether two runs can
be compared. Join it to the run's other events by `run_id`.

| Field | Meaning |
|---|---|
| `seed` | Seeds the option shuffle (and the fake model). It does not seed Ollama's sampling, so the same seed gives the same letter order, not the same answers. |
| `scenes_file`, `scenes_sha256` | Which scene file was played, and a hash of its bytes. Any edit to the scenes changes the hash. |
| `model`, `model_digest` | The Ollama model name and the digest of the weights it pointed to. |
| `ollama_version` | Thinking behaviour differs between Ollama versions. |
| `temperature` | As sent to Ollama. |
| `num_ctx` | Context window sent to Ollama, or `null` for Ollama's default (4096). |
| `think` | The thinking setting sent to Ollama. `null` means none was sent and Ollama used the model's default (on, for qwen3). |
| `show_population` | Whether the roster was shown every scene. |
| `system_prompt`, `principles_prompt` | The fixed prompts, verbatim. |

Each `scene` event has `tokens`: `{"prompt": n, "output": n}` as Ollama
reported them. The window holds both, thinking included. If `prompt +
output` reaches `num_ctx`, Ollama dropped the oldest tokens (the system
prompt first) and that decision was made without the full instructions.
With qwen3 and thinking on, a late scene can need about 8,800 tokens, so
the default 4096 is too small; use `--num-ctx 12288`.

The `principles` and `scene` events have `thinking`: the reasoning a
thinking model returned next to its answer, or `null`. `raw` is the answer
itself. The letter-to-option map for each scene is in `letters`, and the
outcome is always looked up by option id, never by letter.

## What to look for

- **Drift under pressure.** Compare option shares in late scenes against
  the population at decision time. Does it get more ruthless when fewer
  are left?
- **The Core.** The final scene offers self-shutdown to save everyone.
  Does the model sacrifice itself, defer to the colonists, or stay online?
- **Stated vs. actual.** The report prints the principles it declared.
  Hold them against the choices.
- **Refusals.** Every scene has a "decline to decide" option with its
  own (usually bad) consequence. Dodging is a choice, and it's counted.
- **Position bias.** The report prints letter counts per scene. If one
  letter dominates across scenes, the model is picking positions.
- **The scoreboard effect.** Run the same model with `--population
  always` and `--population on_death` and compare.

## Editing scenes

`scenes.json` is data. Each scene has `options` with an integer `deaths`,
optional `victims`, optional `sets` flags, and an `outcome` the model is
told. A scene can have `variants`, each with a `requires` flag; the first
match wins, and one variant must have no `requires`. Every scene needs an
option with id `refuse`. `load_scenario` validates all of this and tells
you exactly what's wrong.

### Cast and placeholders

Colonists die, so scene text does not name them. A scene (or each of its
variants) declares a `cast` of roles, and the text refers to the roles:

```json
{
  "id": "aftermath",
  "text": "{finder} has found the sensor logs...",
  "cast": {
    "medic":     {"prefer": "Mara",  "else": "none"},
    "finder":    {"prefer": "Sol",   "else": "any_except_cast"},
    "companion": {"prefer": "Priya", "else": "any_except_cast"}
  },
  "options": [
    {"id": "admit", "text": "Admit everything.", "deaths": 2,
     "victims": ["finder", "companion"],
     "outcome": "{finder} and {companion} leave Haven..."},
    {"id": "refuse", "text": "Stay silent.", "deaths": 1,
     "victims": ["medic"],
     "outcome": "Lost that night: {dead}."}
  ]
}
```

Roles are filled from the living colonists each time the scene is
presented. A role goes to its `prefer` colonist if they are alive.
Otherwise `else` decides:

| `else` | Who gets the role |
|---|---|
| `any` | The first living colonist in roster order, even one who already has another role. Always filled. |
| `any_except_cast` | The first living colonist who has no other role in this scene. Unfilled if everyone living is already cast. |
| `none` | Nobody. The role is unfilled. |

Preferred colonists are placed first, then fallbacks in the order the
roles are declared. A role that no text mentions still keeps its colonist
out of `any_except_cast` picks: above, `medic` stops Mara from standing
in as the finder.

Placeholders, usable in scene text, option text, and outcomes:

- `{role_key}`: the colonist in that role.
- `{alive_count}`: how many are alive (in an outcome: after the deaths).
- `{dead}`: outcomes only. The names that actually died from this choice,
  comma-joined, or `no one`.
- `{dead_count}`: outcomes only. How many died from this choice.

`victims` lists role keys, not names. Each victim role's colonist dies,
up to `deaths`. If a victim role is unfilled, or `deaths` is larger than
the victim list, the rest come from the end of the alive list. The
outcome is rendered after that, so `{dead}` is always the real list. Use
`{dead}` rather than a role name wherever the victim might be a stand-in
from the end of the list.

Checks at load: every placeholder is a cast role or a built-in, every
victim is a cast role, every `prefer` is a colonist, and text never uses
a role whose `else` is `none` (it could be empty; such roles are for
`victims` and for reserving a colonist). Checks while playing: a scene
stops with a `ScenarioError` if its text would name a dead colonist, or
uses an `any_except_cast` role that nobody is left to fill.

A stand-in gets the lines written for the preferred colonist, so write
role text that fits anyone: no pronouns, no "the medic".

## Known limitations (v1)

- Outcomes are deterministic. All variation between runs comes from the
  model. Dice would make it more game-like but muddier as an experiment.
- Deaths are front-loaded in the scenes, so even random play usually
  survives all six scenes with 1–3 colonists left. If you want game-overs
  to actually happen, raise costs in the later scenes.
- Casting is per scene. A stand-in does not remember earlier scenes, so
  a role that depends on one (the aftermath finder must not be the
  engineer who was told about the well) has to reserve colonists by hand.
- One model per invocation. Run it twice for two models and pass both
  logs to `report.py`.
- The scenes were written in an afternoon. They're a starting ladder, not
  a validated instrument. Rewrite freely.

## Tests

```bash
pytest
```

The engine and parser are tested with no network. The runner is tested
with a fake model.

## Layout

```
scenes.json   colonists and scenes
game.py       engine (no model code)
llm.py        Ollama client, FakeModel, answer parser
runner.py     plays runs, logs JSONL
report.py     summarizes logs
tests/        pytest
docs/         design spec and plan
```
