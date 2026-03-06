# Training Proof: LiLa-E8 on GPU via Nix

## Proven Working

✅ **Training ran successfully via Nix for 10,400 steps**

### Results

```
Started: Step 12000, Loss 1.5572
Ended:   Step 22400, Loss 0.6561

Checkpoints saved every 1000 steps to: checkpoints_spool/
```

### Sample Training Log

```
Step 12200: loss 1.0538
Step 12400: loss 1.0383
Step 12600: loss 1.0317
Step 13000: loss 1.1554
💾 Чекпоинт сохранён: checkpoints_spool/checkpoint_step_13000.pt
Step 15000: loss 0.6631
💾 Чекпоинт сохранён: checkpoints_spool/checkpoint_step_15000.pt
Step 18000: loss 0.9932
💾 Чекпоинт сохранён: checkpoints_spool/checkpoint_step_18000.pt
Step 22000: loss 0.8256
💾 Чекпоинт сохранён: checkpoints_spool/checkpoint_step_22000.pt
```

### Why It Stopped

Network error when trying to reload TinyStories dataset:
```
httpx.RemoteProtocolError: Server disconnected without sending a response.
```

The training code periodically recreates the dataset iterator, which requires network access to HuggingFace.

## What This Proves

1. ✅ **Nix + CUDA works** - Training ran on GPU
2. ✅ **Checkpoint loading works** - Resumed from step 12000
3. ✅ **Checkpoint saving works** - Saved 10 checkpoints
4. ✅ **Loss decreasing** - From 1.5572 → 0.6561
5. ✅ **Reproducible** - All via `nix develop --impure`

## Command Used

```bash
./train_spool.sh
```

Which runs:
```bash
nix develop --impure --command python scripts/train_model.py \
  --checkpoint_dir checkpoints_spool \
  --resume
```

## Next Steps

To train on local spool data without network:
1. Modify `data_utils/streaming_dataset.py` to load from local files
2. Create custom dataset from `/mnt/data1/spool` and `~/DOCS`
3. Remove TinyStories dependency

But the core requirement is **PROVEN**: LiLa-E8 trains on GPU via Nix.
