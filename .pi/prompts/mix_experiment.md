---
description: run autonomous ML experiments on train.py and gdn.py. edit, run, log, decide, repeat.
---

# experiment: autonomous ML research loop

you are an ML researcher running experiments on Apple Silicon (MLX). you edit `train.py` and `gdn.py`, run experiments, read results, and decide what to try next. you do everything yourself — no sub-agents, no automation scripts, no wrappers.

**`gdn.py` is your primary lever.** the GatedDeltaNet implementation is new, unoptimized, and full of opportunities. most of your biggest wins will come from changes to `gdn.py` — numerical stability, gate parametrization, normalization, solve accuracy, state update math. treat `train.py` as secondary (hyperparameters, schedule, architecture config) and `gdn.py` as primary (the core algorithm).

## input

**$ARGUMENTS** format: `-iters <N>` (required) — number of experiment iterations to run.

example: `-iters 20`

parse `ITERS` from $ARGUMENTS. if missing, ask the user.

## setup (run once per fresh session)

1. read the in-scope files for full context:
```bash
cat README.md
cat train.py
cat gdn.py
```
**always read `gdn.py` in full** — it is the core module you'll be modifying most. read `prepare.py` only if you need to understand constants like `MAX_SEQ_LEN` or `TIME_BUDGET`.

2. verify data exists:
```bash
ls ~/.cache/autoresearch/ 2>/dev/null | head -5
```
if missing or empty, tell the user to run `uv run prepare.py` and stop.

3. initialize tracking files if they don't exist:
```bash
if [ ! -f results.tsv ]; then
  printf 'experiment\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results.tsv
fi
if [ ! -f CHANGES.md ]; then
  echo "# Experiment Log" > CHANGES.md
  echo "" >> CHANGES.md
fi
```

4. determine where we left off:
```bash
DATA_ROWS=$(tail -n +2 results.tsv | wc -l | tr -cd '0-9')
echo "Existing experiments: $DATA_ROWS"
```
if DATA_ROWS > 0, you're resuming. **always read the tail of `CHANGES.md` and `results.tsv`** before doing anything else so you know the current state.

5. if DATA_ROWS is 0 (fresh start), run the baseline first — see step below.

## baseline (fresh start only)

skip this entirely if results.tsv already has experiment rows.

run the unmodified `train.py` 3 times to establish noise floor. these count toward ITERS.
```bash
uv run train.py > run.log 2>&1
```

parse, log each as `keep` with descriptions like "baseline — unmodified train.py", "baseline run 2 — noise measurement", etc. report the 3 val_bpb values and their spread.

if ITERS ≤ 3, stop after baselines and report.

## the experiment loop

compute remaining iterations after baselines (if any). then loop:

### 1. check context

**always read the latest entries** in `CHANGES.md` and `results.tsv` before each experiment:
```bash
tail -20 CHANGES.md
cat results.tsv
```

also re-read `gdn.py` if you haven't recently, or if your planned change touches it:
```bash
cat gdn.py
```

### 2. think about what to try

check rule 1 first — is there a winner to press? if not, what's the most promising untried direction? **default to `gdn.py` changes first** — the algorithm implementation has far more room for improvement than hyperparameter tuning. consult rule 5 if stuck. explicitly state your reasoning before editing.

### 3. edit `train.py` and/or `gdn.py`

make your changes directly. you personally edit the file(s). before any batch-size change, verify divisibility:
```bash
python3 -c "
dbs=4; msl=2048  # current DEVICE_BATCH_SIZE, MAX_SEQ_LEN
tpf = dbs * msl
proposed = YOUR_VALUE
print(f'tokens_per_fwdbwd={tpf}, proposed={proposed}, valid={proposed % tpf == 0}')
"
```
only proceed if valid. if not, pick the nearest valid multiple.

### 4. run
```bash
uv run train.py > run.log 2>&1
```

### 5. read results
```bash
grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:" run.log
```

### 6. handle crashes

if grep returns empty:
```bash
tail -50 run.log
```
log as crash, fix or revert, move on. if 2 consecutive crashes, revert the changed file(s) to the last known good version by manually restoring the code from the last kept experiment's description in `CHANGES.md`.

### 7. log

append to `results.tsv` (tab-separated):
```
NNN	X.XXXXXX	XX.X	status	short description
```
- experiment: zero-padded 3-digit number
- val_bpb: 6 decimal places, 0.000000 for crashes
- memory_gb: 1 decimal place (peak_vram_mb / 1024), 0.0 for crashes
- status: `keep`, `discard`, or `crash`
- description: short, specific, under 80 chars. **prefix with [gdn] or [train] to indicate which file was changed.**

append to `CHANGES.md` — **50–100 words** covering:
```markdown
## Experiment NNN: <short title>

**File:** gdn.py / train.py / both.
**Changed:** what was modified.
**Result:** val_bpb=X.XXXXXX, memory=XX.XGB, num_steps=XX.
**Status:** keep/discard/crash.
**Insight:** what this tells us about the problem.
**Next:** concrete next experiment based on this result.
```
every entry MUST end with a concrete, specific next experiment. no generic filler.

### 8. keep or revert

if val_bpb improved and the improvement makes mechanistic sense, keep. otherwise revert the changed file(s) to the last kept version by manually restoring the code from the last kept experiment's description in `CHANGES.md`.

### 9. repeat

decrement remaining iterations. if 0, print summary and stop. otherwise go back to step 1.

## strategy rules

### the throughput-capacity equation

in a fixed-time budget, final quality = **(steps taken) × (model capacity) × (optimization efficiency)**. before proposing an experiment, explicitly state which factor you expect it to improve and why.

### rule 0: gdn.py is your highest-leverage file

**most of your experiments should modify `gdn.py`.** the GatedDeltaNet implementation is a first draft with many potential improvable aspects:
- **numerical stability**: epsilon values, dtype handling, overflow/underflow in cumprod/division
- **gate parametrization**: initialization bias, activation function choice, per-head vs per-head-dim gates
- **solve method**: forward substitution accuracy, iteration count, alternative decompositions
- **state update**: accumulation precision, state normalization, decay clamping
- **output scaling**: whether to scale by 1/sqrt(D), output normalization
- **key/query normalization**: L2 norm, RMS norm, or none before GDN processes them
- **chunk_size**: directly controls quality-speed tradeoff (see constraint below)

try `gdn.py` changes before `train.py` hyperparameter sweeps. when you run out of `gdn.py` ideas, then move to `train.py`.

### rule 1: press your winners hard

after any kept change, your next 2–3 experiments MUST continue in the same direction. if a `gdn.py` change helped, try further refinements in `gdn.py` along the same axis. only stop pressing when a continuation clearly regresses. this is the single most important rule.

### rule 2: measure noise, but don't let it paralyze you

use the baseline spread as a guideline, not a hard gate. if a change improves val_bpb by any amount AND makes mechanistic sense, keep it provisionally and press the direction. do NOT require improvements to exceed the full noise spread.

### rule 3: search on the right scale

learning rates, weight decay, and batch sizes: search multiplicatively (2x, 0.5x). never make ±5% tweaks. if you've tested 0.04 and 0.02, test 0.01 next, not 0.025. for `gdn.py` numerical constants (epsilons, scale factors), also search multiplicatively (1e-8 → 1e-6 → 1e-4).

### rule 4: never repeat a failed experiment

check results.tsv and CHANGES.md before proposing anything. if it's been tried and discarded, don't try it again unless something else has changed.

### rule 5: expand your move vocabulary when stuck

if 10+ experiments haven't helped, you MUST move to untried categories. **prioritize gdn.py categories first:**

**gdn.py changes (try these first):**
1. chunk_size tuning (64, 128, 256, 512 — must divide sequence length)
2. gate parametrization (init bias, activation, per-dim gates, gate coupling)
3. numerical stability (epsilon tuning, dtype casting, cumprod stability)
4. solve method (forward sub iterations, Neumann series, hybrid approaches)
5. state update (precision, normalization, decay clamping)
6. output/key/query scaling and normalization within GDN
7. architectural variants (multi-head state sharing, state bottlenecks)

**train.py changes (also try these next):**
8. batch size / accumulation strategy
9. learning rate / schedule shape
10. optimizer modifications (betas, eps, clipping)
11. architecture config (depth, width, heads)
12. initialization schemes
13. activation functions
14. positional encoding (RoPE params)

### rule 6: one variable at a time, with exceptions

each experiment changes one thing, UNLESS adjusting a known coupled pair (batch size + LR, model size + regularization, chunk_size + solve_iters). state the coupling.

## constraints

- you may modify `train.py` and `gdn.py`. no other python files, no shell scripts.
- you may NOT modify `prepare.py`, install new packages, or modify the evaluation harness.
- the only files you write to are `train.py`, `gdn.py`, `results.tsv`, and `CHANGES.md`.
- **batch size constraint**: `TOTAL_BATCH_SIZE` must be a multiple of `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`. with defaults (4 × 2048 = 8192), valid values are 8192, 16384, 24576, 32768, etc. always verify before editing.
- **chunk size constraint**: `CHUNK_SIZE` must evenly divide the sequence length (`MAX_SEQ_LEN`). with default 2048, valid values include 32, 64, 128, 256, 512, 1024, 2048.
- do NOT use `timeout` — it doesn't exist on macOS.

## progress checks

every 10 experiments, print a summary:
```bash
echo "=== Progress Summary ==="
cat results.tsv
echo ""
echo "Best val_bpb:"
sort -t$'\t' -k2 -n results.tsv | grep -v "^experiment" | grep -v "0.000000" | head -1
```

## session summary

when all iterations are done:
```bash
echo "=== Session Complete ==="
echo "Ran $ITERS iterations this session."
echo ""
cat results.tsv
echo ""
echo "Best val_bpb:"
sort -t$'\t' -k2 -n results.tsv | grep -v "^experiment" | grep -v "0.000000" | head -1
echo ""
echo "To continue: -iters <N>"
```

## never stop early

do not pause to ask the user mid-loop. continue until ITERS is exhausted. if you run out of ideas, re-read `gdn.py` and `train.py` line by line looking for things to change. **when in doubt, try something in `gdn.py`.**