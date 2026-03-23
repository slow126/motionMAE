# Flow MAE Experiment

Isolated masked autoencoder experiment for FlyingThings-only pretraining.

## Current Run Definition

- Dataset: FlyingThings only
- Input channels per sample: RGB source (3) + RGB target (3) + sparse flow (2) + validity mask (1)
- Image size: 256x256
- Patch size: 16x16
- Masking: 75% of valid flow patches, RGB always visible
- Encoder: 6-layer ViT-style transformer, width 384
- Decoder: 2-layer transformer, width 256
- Target: dense flow only
- Loss: Smooth L1 on valid pixels only
- Logging: TensorBoard + epoch checkpoints
- Budget: one GPU, one 24-hour Slurm run

## Train

```bash
python3 scripts/train_flow_mae.py --config src/configs/flow_mae/flyingthings_vits_rc.yaml
```

## Submit On RC

```bash
export FLOW_MAE_CONDA_ENV=cuda
sbatch slurm/run_flow_mae_flyingthings_rc.sbatch
```

Optional overrides:

```bash
FLOW_MAE_CONFIG=/path/to/other_config.yaml sbatch slurm/run_flow_mae_flyingthings_rc.sbatch
```

## Outputs

Each run writes to `snapshots/flyingthings_flow_mae_vits_<timestamp>/` with:

- `config.yaml`
- `launch.txt`
- `checkpoints/`
- `tensorboard/`
