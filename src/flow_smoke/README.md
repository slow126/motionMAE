# Flow smoke test (Point Odyssey)

This folder contains a minimal prototype for:

- conditional deterministic flow prediction `p(flow | image, dt)` and
- a small conditional VAE baseline.

## Files

- `dataset.py`: Point Odyssey pair dataset that emits `src_img`, `flow`, `dt`, and `valid_flow_mask`.
- `models.py`: 
  - `DeterministicUNet`
  - `ConditionalFlowVAE`
- `train_flow_smoke.py`: lightweight training/eval loop.
- `__init__.py`: re-exports.

## Quick run

```bash
python src/flow_smoke/train_flow_smoke.py \
  --manifest-path /path/to/manifest.jsonl \
  --pointodyssey-root /path/to/pointodyssey_root \
  --model det \
  --dt-values 1,2,3,4 \
  --epochs 12 \
  --batch-size 4 \
  --size 256
```

Switch `--model vae` and optionally tune `--z-dim` and `--beta-*`.

## Minimal ablations for phase-1

- fixed dt: `--dt-values 2`
- variable dt: `--dt-values 1,2,3,4`
- baseline vs conditional VAE:
  - `--model det`
  - `--model vae`
- image only vs grayscale:
  - default color
  - `--use-grayscale`

Logs/checkpoints are written to:
`--output-dir/<exp-name>` (`det_dt_1_2_3_4` by default).

