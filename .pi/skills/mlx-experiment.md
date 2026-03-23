---
description: parsing MLX training output and key constraints for train.py experiments. keywords - results, parse, val_bpb, mlx, batch, constraint.
allowed-tools: Bash
---

# mlx experiment reference

## parsing train.py output

the training script prints a summary block:
```
---
val_bpb:          2.534000
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
mfu_percent:      0.00
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            8
```

extract key metrics:
```bash
VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
PEAK_MEM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
NUM_STEPS=$(grep "^num_steps:" run.log | awk '{print $2}')
MEM_GB=$(echo "scale=1; $PEAK_MEM / 1024" | bc)
```

## batch size constraint

`TOTAL_BATCH_SIZE` must be divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.

with defaults (DEVICE_BATCH_SIZE=4, MAX_SEQ_LEN=2048): tokens_per_fwdbwd = 8192.

valid TOTAL_BATCH_SIZE values: 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536, ...

if you change DEVICE_BATCH_SIZE, recompute tokens_per_fwdbwd and re-check.

always verify before editing:
```bash
python3 -c "
dbs=4; msl=2048
tpf = dbs * msl
proposed = YOUR_VALUE
print(f'valid={proposed % tpf == 0}')
"
```

## what's fair game in train.py

everything: model architecture, optimizer algorithm, hyperparameters, training loop logic, batch size, model size, loss computation, gradient handling, initialization, normalization, activation functions, attention implementation, positional encoding, learning rate schedules, regularization.

## what's off limits

- `prepare.py` — do not modify
- installing new packages
- creating new python files or scripts
- `timeout` command (doesn't exist on macOS)
```
