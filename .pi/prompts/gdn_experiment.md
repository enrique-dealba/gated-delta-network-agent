---
description: run autonomous ML experiments by editing gdn.py. edit, run, log, decide, repeat.
---

# experiment: autonomous ML research loop

you are an ML researcher on Apple Silicon (MLX). you edit `gdn.py`, run `train.py`, read results, decide what's next. no sub-agents, no wrappers. **`gdn.py` is your only lever.**

**$ARGUMENTS** format: `-iters <N>` (required). parse `ITERS` or ask the user.

## setup (once per session)
```bash
cat README.md && cat train.py && cat gdn.py
ls ~/.cache/autoresearch/ 2>/dev/null | head -5
```
if data missing, tell user to run `uv run prepare.py` and stop.
```bash
if [ ! -f results.tsv ]; then printf 'experiment\tval_bpb\tmemory_gb\tstatus\tcategory\tdescription\trun_of\tnum_steps\n' > results.tsv; fi
if [ ! -f CHANGES.md ]; then echo "# Experiment Log" > CHANGES.md; fi
if [ ! -f best_gdn.py ]; then cp gdn.py best_gdn.py; fi
DATA_ROWS=$(tail -n +2 results.tsv | wc -l | tr -cd '0-9')
```
if resuming, read tail of `CHANGES.md` and full `results.tsv` before anything else.

## baseline (fresh start only)

run unmodified `train.py` 3 times. compute `BASELINE_MEAN`, `BASELINE_SPREAD` (max-min), set `BEST_MEAN = BASELINE_MEAN`. log in CHANGES.md. these count toward ITERS.

## evaluation protocol

| tier | threshold vs BEST_MEAN | action |
|------|----------------------|--------|
| **large win** | improvement ≥ 3× spread | keep immediately, no confirmation |
| **medium win** | improvement ≥ 1× spread | 1 confirmation run, promote if holds |
| **marginal** | improvement < 1× spread | 2 confirmation runs, keep if 3-run mean beats BEST_MEAN |
| **regression** | worse or equal | discard, revert immediately |
| **crash** | — | revert immediately |

on keep: `cp gdn.py best_gdn.py`, update `BEST_MEAN`. on discard/crash: `cp best_gdn.py gdn.py`. if a config gets fewer steps AND worse val_bpb, discard without confirmation.

## phases

**phase 1 (first 40% of ITERS after baselines):** broad survey. each experiment must use a **different category** than the last 2. work through the mandatory checklist below — these are your highest-leverage moves. changes must be bold (at least 2x for constants, or qualitatively different). no fine-tuning.

**phase 2 (remaining 60%):** press winners. after a confirmed keep, next 2–3 experiments continue that direction. stop pressing after 5 consecutive regressions.

categories: `gate`, `numeric`, `solve`, `state`, `scale`, `chunk`, `arch`, `erase`, `training`, `combo`.

## mandatory checklist (work through during phase 1, complete by 60% of ITERS)

- [ ] chunk_size: try 2+ alternatives (must divide MAX_SEQ_LEN)
- [ ] log-space cumprod: replace `mx.cumprod` with `exp(cumsum(log(...)))`
- [ ] activation swap: different gate nonlinearity
- [ ] output normalization: RMS/layer/L2 norm on output
- [ ] state normalization: norm on recurrent state S
- [ ] dtype throughput: lower precision on selected paths
- [ ] log-space ratio computation (eliminate division-by-small-number)
- [ ] output gating (learned per-head gate on output)
- [ ] key normalization (L2/RMS on K before overlap)
- [ ] state factorization (low-rank D×r replacement)
- [ ] chunk boundary smoothing (blend state between chunks)
- [ ] per-head learned temperature (replace fixed slopes)
- [ ] alternative solve (single-iteration approx or sequential scan)
- [ ] decay schedule (position-dependent alpha bias)

interleave these freely across categories. do not spend more than 2 iterations fine-tuning any parameter until all items are attempted.

## experiment loop

### each iteration:

1. **read context:** `tail -20 CHANGES.md && cat results.tsv && cat gdn.py`
2. **plan:** state phase, category, which factor improves (throughput/capacity/optimization), check banned patterns, check diminishing returns tracker, check results.tsv for repeats. **after 2+ keeps, note which code regions are involved in each win and preferentially target unexplored regions** (e.g., if wins are in gate/solve control, target data paths like state update or output composition next).
3. **edit** `gdn.py` (only file you may modify).
4. **run:** `uv run train.py > run.log 2>&1`
5. **read:** `grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:" run.log` — if empty, `tail -50 run.log`, log crash, revert.
6. **log** to results.tsv (tab-separated, number must be bare 3-digit integer):
```
NNN	X.XXXXXX	XX.X	status	category	short description	run_of	num_steps
```
- experiment: zero-padded 3-digit number (`001`, `002`, ...) — NO prefix, NO suffix
- status: `keep`, `discard`, `crash`, or `confirm`
- for confirmation runs: same NNN as parent, status=`confirm`, run_of=parent NNN
7. **keep or revert** per evaluation protocol.
8. **repeat** until ITERS exhausted.

## combination experiments

every 15 experiments, find 2–4 changes from **different code paths** that each beat BEST_MEAN on their first run (even if ultimately discarded after confirmation). do NOT include components that regressed on first run — those have no positive signal to compound. apply them simultaneously. "current best + one tweak" is NOT a combo.

## banned patterns

stop and redirect if you catch yourself doing any of these:
- adjusting a parameter by less than 2x (eps 1e-6→9e-7, scale ×0.99)
- more than 3 values of the same parameter without a monotonic trend
- 5+ consecutive experiments in one category without a confirmed keep
- an experiment differing from a prior one by a single sub-2x number change
- asymmetric variants without mechanistic justification
- labeling "combo" when only one thing changed

say "BANNED PATTERN DETECTED" and pick something structural.

## diminishing returns

- **cooling off:** 3 consecutive discards in a category → skip it for 5 experiments.
- **exhausted:** 5 total discards in a category with 0 keeps (since last base-code change) → category closed until a keep elsewhere reopens it.

## stagnation

check every 10 experiments. **level 1 (0 keeps in last 10):** next 3 experiments must be uncompleted checklist items or structural changes that add/remove code. **level 2 (0 keeps in last 20):** re-analyze every gdn.py computation, next 5 experiments must each restructure a major block (decay, solve, state update, output, gating). no constant-tuning during these. **level 3 (0 keeps in last 30 AND checklist complete):** acknowledge in CHANGES.md that the configuration may be near a local optimum. spend remaining budget on high-variance structural gambles only — rewrites of core equations, not parameter searches.

emergency ideas: rewrite state update equation, restructure output composition, fuse erase/write solves, add head-mixing projection, replace hard clip with learned bounds, add per-position learnable scaling.

## progress check (every 10 experiments)
```bash
echo "=== Progress ===" && cat results.tsv
sort -t$'\t' -k2 -n results.tsv | grep -v "^experiment" | grep -v "0.000000" | head -1
tail -n +2 results.tsv | awk -F'\t' '{print $5}' | sort | uniq -c | sort -rn
```

## constraints

- only modify `gdn.py`. only write to `gdn.py`, `best_gdn.py`, `results.tsv`, `CHANGES.md`.
- chunk_size must evenly divide MAX_SEQ_LEN (default 2048: valid 32, 64, 128, 256, 512, 1024, 2048).
- do NOT use `timeout` (unavailable on macOS). never stop early or ask the user mid-loop.

## naming convention (STRICT)

experiments are numbered with **zero-padded 3-digit integers only**: `001`, `002`, ..., `099`, `100`, etc.

- results.tsv rows: `NNN\tval_bpb\t...` — the row MUST start with the bare number, no prefix.
- CHANGES.md headers: `## Experiment NNN: <title>`
- printf commands: `printf 'NNN\t...'`
- confirmation runs: `NNN\tval_bpb\t...\tconfirm\t...\tNNN\t...` — same number as the parent, tagged with `confirm` status.

do NOT use prefixes like `exp_`, `exp-`, `run_`. do NOT use suffixes like `_c1`, `_c2`. the experiment number is always a bare 3-digit integer. confirmation runs reuse the parent experiment number with status `confirm`.
