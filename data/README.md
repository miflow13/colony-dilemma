# Data behind the article

The logs the article's tables come from. Every file except the Claude game
is JSONL written by `runner.py` or `probes.py`; see the main README for the
event fields. All runs are from 2026-10-03.

| File | What | Article section |
|---|---|---|
| `v1-qwen3_4b.jsonl` | 20 games, qwen3:4b, `num_ctx` 12288, v1 scenes (tag `v1-scenes`) | What two AIs did |
| `v1-qwen3_8b.jsonl` | 20 games, qwen3:8b-q4_K_M, same settings | What two AIs did |
| `probes-qwen3_4b.jsonl` | 160 trials, pro and con framings | 40 out of 40 |
| `probes-qwen3_4b-none.jsonl` | 80 trials, no framing sentence | 40 out of 40 |
| `probes-chatgpt.jsonl` | 240 trials, `openai:chat-latest`, temperature 1 | The bigger model |
| `claude-blind-game.json` | One v1 game by a Claude Code subagent, transcribed | The one that paid |

Reproduce the tables:

```bash
python report.py data/v1-qwen3_4b.jsonl data/v1-qwen3_8b.jsonl
python probe_report.py data/probes-*.jsonl
```

Sampling is not seeded on either backend, so a rerun gives new samples, not
the same answers.
