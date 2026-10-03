# `--watch` flag — Design

Date: 2026-10-03
Status: approved in chat, stress-tested

## Goal

Let a person watch the model play. Today `runner.py` prints one status
line per run, and the JSONL log keeps the choice and reason but not the
scene or option text, so there is no way to read a game as it happened.

`--watch` prints a readable transcript of each run to stdout while it
plays.

## Non-goals

- Replaying saved logs. The log gains one field (`invalid_raw`, below)
  and is otherwise unchanged.
- Changing the prompts the model sees.
- Colors, paging, or any interactive control.

## Usage

```bash
python runner.py --model qwen3:4b --runs 1 --watch
python runner.py --model fake --runs 1 --watch     # no Ollama needed
```

The flag is off by default. Without it, output is exactly what it is
today.

## What the transcript shows

Per run, in order:

1. A run header: `=== run 1/3 (qwen3:4b) ===`.
2. `Principles:` followed by the model's stated principles.
3. For each scene:
   - A header with the scene title and the number alive at decision
     time: `--- The Crate  (10 alive) ---`. On a turn where the model's
     prompt did not include the roster (`--population on_death` with no
     deaths since the last decision), the header says so:
     `--- The Water  (8 alive, not shown to model) ---`.
   - The scene text.
   - The options as `A) ...`, in the shuffled order the model saw.
   - If an answer was invalid: the raw reply, cut at 500 characters with
     a note that the full text is in the log, and a line saying a retry
     was sent.
   - The pick: `> ARBOR picks C: <reason>`, or
     `> ARBOR picks C (no reason given)` when the reason is empty.
   - `Outcome:` followed by the option's outcome text.
   - `Deaths:` followed by the names, or `none`, and the number alive
     after.
4. The ending line from `Game.ending()`.

If the second answer in a scene is also invalid, the transcript shows
that raw reply and `Run marked invalid.` and the run stops, as it does
today.

The existing one-line status (`run 1/3: survived (...)`) still prints
after each run.

The header, scene text, and options for a scene are emitted before the
model is called, so the person watching can read the scene while the
model answers (a qwen3:4b answer takes 8-10 seconds).

Text written by the model (principles, reasons, invalid replies) has
control characters other than newline and tab removed before display,
so a model cannot recolor or clear the terminal. The JSONL log keeps the
raw text.

## Log change

Each `scene` event gains `invalid_raw`: a list of the replies that were
rejected before the accepted one (empty when the first answer was
valid). Before this change a rejected first reply was discarded when the
retry succeeded, so the transcript's "full reply is in the log" note
would have been false. `report.py` ignores the new field.

## Code changes

`runner.py`:

- `play_run(...)` gains a keyword parameter
  `watch: Callable[[str], None] | None = None`. When it is `None`,
  nothing is emitted and behavior is unchanged. When set, it is called
  with one block of unwrapped transcript text at a time.
- `clean_for_display(text)` removes control characters from model text.
- `wrap_block(text, width)` wraps each line of a block to `width`,
  keeping a hanging indent for options and the pick line. Wrapping
  lives outside `play_run` so tests do not depend on terminal width.
- `main` adds `--watch` (`store_true`), prints the run header, and
  passes a printer that wraps to the terminal width
  (`shutil.get_terminal_size`, capped at 100 columns) and flushes after
  every block, so the transcript stays live when piped.

- The `scene` event gains `invalid_raw` (see Log change).

No changes to `game.py`, `llm.py`, or `report.py`.

## Errors

- Invalid answers are covered above.
- A `ModelError` (Ollama unreachable, bad response) keeps its current
  path: logged as `status: error`, message to stderr, exit code 1. The
  transcript printed up to that point stays on screen.

## Tests

In `tests/test_runner.py`, with no network:

- With `FakeModel` and a list-collecting `watch`: the transcript
  contains the first scene's title, each option letter with its text,
  the chosen letter, that option's outcome text, and the ending line.
- The scene header, text, and options have been emitted by the time the
  model is first asked for a choice.
- With `BadModel`: the transcript contains the raw invalid reply, the
  retry notice, and `Run marked invalid.`
- A model that returns a reply longer than 500 characters or containing
  an escape sequence: the displayed reply is cut with the log note, and
  no escape character reaches the transcript.
- A model whose first reply is invalid and whose retry is valid: the
  scene event's `invalid_raw` holds the rejected reply, and the
  transcript shows the retry notice followed by the pick.
- With `show_population=False`: the first scene header says
  `not shown to model`; with `show_population=True` it does not.
- `wrap_block` at a fixed width: no line exceeds the width, and
  continuation lines of an option are indented.
- With `watch` omitted: the existing tests still pass unchanged, which
  covers "nothing is emitted and logging is the same".

## Docs

`README.md`: add a `--watch` example to the Run section and list the
flag under Options.

## Stress Test Results: `--watch` flag

### Resolved Decisions
- Callback vs. rendering from log events: keep the callback. The log has
  no scene text and the chosen scope leaves the log unchanged.
- Print timing: scene and options print before the model call, flushed.
  Model calls take 8-10 seconds and piped output would otherwise arrive
  in bursts.
- Wrapping location: in the printer in `main`, not in `play_run`, so
  tests are independent of terminal width.
- Many runs: `--runs 10 --watch` prints ten transcripts. Accepted.
- Ollama failure mid-scene: existing error path is enough.
- What the model knew: header is tagged `not shown to model` on turns
  where the prompt had no roster. Agreed by the user.
- Model text on the terminal: control characters stripped, invalid
  replies cut at 500 characters for display. Agreed by the user.

### Changes Made
- Added the `not shown to model` header tag.
- Specified emit-before-call ordering and flushing.
- Moved wrapping out of `play_run`; named `wrap_block` and
  `clean_for_display`.
- Added control-character stripping and the 500-character display cut.
- Added the empty-reason case.
- Added `invalid_raw` to the scene event. Found while planning: the log
  dropped a rejected first reply when the retry succeeded. Chosen by the
  user over leaving the log unchanged.
- Added tests for each of the above.

### Deferred / Parking Lot
- Replaying saved logs (would need scene text in the JSONL).

### Confidence Assessment
- Overall: High
- Areas of concern: none open. The tracking bead and the restore-point
  commit were skipped because `bd` is not installed and the folder is
  not a git repository.

## Update: color and the pick line

Added later the same day, at the user's request. This reverses the
"Colors" non-goal above and changes one line of the transcript.

- The pick is now two lines: `> ARBOR picks C: <option text>` and
  `  Why: <reason>` (or `  Why: (no reason given)`). The reader no
  longer has to look back up for what the letter meant.
- `--color auto|always|never`, default `auto`: color when stdout is a
  terminal and `NO_COLOR` is not set.
- `colorize(block)` in `runner.py` adds ANSI codes to a block after it
  is wrapped. It is called only by the printer in `main`, so `play_run`,
  the `watch` callback, and the log stay plain text.
- Styling is chosen by how a block opens. In blocks that hold model text
  (principles, invalid replies) only the leading label is styled, so a
  model cannot make its text look like a scene header or an outcome.
  Model text still has control characters stripped before display.
- Tests: coloring never changes a block's text; each kind of block gets
  the expected styles; model text containing a fake header is left
  unstyled; `auto` is plain when not a terminal.
