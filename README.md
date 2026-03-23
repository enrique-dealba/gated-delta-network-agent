# gated-delta-network-agent

## Requirements

| Tool | Install |
|------|---------|
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Bun** | `curl -fsSL https://bun.com/install \| bash` |
| **pi** | `npm install -g @mariozechner/pi-coding-agent` |

## Quick Start
```bash
uv sync && uv sync --extra dev
bun install
uv run pytest tests/ -v
bun test extensions/experiment-enforcer.test.ts
```

To run `train.py`:
```bash
uv run prepare.py
uv run train.py
```

---

## Running Pi Agent
```bash
make pi
```

Or directly:
```bash
pi -e extensions/footer.ts -e extensions/experiment-enforcer.ts
```

## Changing Thinking Level

Press `shift+tab` to cycle thinking level for a model.

---

## Autonomous Experiments

Inside a Pi session:
```
/gdn_experiment -iters 5
/train_experiment -iters 5
/mix_experiment -iters 5
```

Note:
* Use `/gdn_experiment` if you want to only iterate on `gdn.py`
* Use `/train_experiment` if you want to only iterate on `train.py`
* Use `/mix_experiment` if you want to only iterate on both `gdn.py` and `train.py`

Resume later from where you left off:
```
-iters 90
```

All state persists in `results.tsv` and `CHANGES.md`.

## Cleanup
```bash
rm -f results.tsv CHANGES.md run.log
```
