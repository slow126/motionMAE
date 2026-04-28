from __future__ import annotations

import torch

from percieverIO_Exp.data.variable_flow import (
    VariableObservationFlowDataConfig,
    build_variable_flow_sample,
    variable_observation_collate,
)
from percieverIO_Exp.lightning import VariableFlowLightningModule
from percieverIO_Exp.model import VariableFlowConfig


def make_sample(phase: int, seed: int = 0) -> dict:
    h = w = 16
    image1 = torch.rand(3, h, w)
    image2 = torch.rand(3, h, w)
    flow = torch.rand(2, h, w)
    valid = torch.ones(h, w, dtype=torch.bool)
    cfg = VariableObservationFlowDataConfig(
        root="/tmp/unused",
        phase=phase,
        image_size=(h, w),
        query_stride=4,
        rgb_patch_size=3,
        batch_size=1,
        val_batch_size=1,
        num_workers=0,
        persistent_workers=False,
        normalize_rgb=False,
        fixed_observed_fraction=0.25,
        fixed_mask_mode="random",
    )
    return build_variable_flow_sample(
        sample_id=f"sample_{phase}_{seed}",
        image1=image1,
        image2=image2,
        flow=flow,
        valid=valid,
        config=cfg,
        rng_seed=seed,
    )


def test_phase_semantics_and_collate() -> None:
    sample0 = make_sample(0, seed=1)
    sample1 = make_sample(1, seed=2)
    sample2 = make_sample(2, seed=3)
    sample3 = make_sample(3, seed=4)
    sample4 = make_sample(4, seed=5)

    assert sample0["view_a"]["mask_type"] == "dense"
    assert torch.all(sample1["view_a"]["tokens"][:, 0] == 0.0)
    assert torch.any(sample1["view_a"]["tokens"][:, 1] == 1.0)
    assert torch.any(sample2["view_a"]["tokens"][:, 0] == 1.0)
    assert sample3["view_b"] is not None
    assert sample4["view_b"] is not None
    assert sample2["query_inputs"].shape[0] == sample2["target_flow_q"].shape[0]

    batch = variable_observation_collate([sample2, make_sample(2, seed=6)])
    assert batch["view_a"]["tokens"].shape[0] == 2
    assert batch["view_a"]["pad_mask"].dtype == torch.bool
    assert batch["query_inputs"].shape[1] == batch["target_flow_q"].shape[1]


def test_model_and_loss_paths_are_finite() -> None:
    batch2 = variable_observation_collate([make_sample(2, seed=7), make_sample(2, seed=8)])
    batch4 = variable_observation_collate([make_sample(4, seed=9), make_sample(4, seed=10)])

    model_cfg = VariableFlowConfig.from_config_dict(
        {
            "image_size": [16, 16],
            "query_stride": 4,
            "num_frequency_bands": 4,
            "input_width": 16,
            "query_width": 16,
            "num_latents": 32,
            "latent_dim": 32,
            "depth": 2,
            "self_attention_heads": 4,
            "cross_attention_heads": 1,
            "dropout": 0.0,
        }
    )
    module = VariableFlowLightningModule(
        model_config=model_cfg,
        training_config={"image_size": [16, 16], "query_stride": 4, "lr": 1e-4, "max_epochs": 1},
    )

    loss2 = module.training_step(batch2, 0)
    loss4 = module.training_step(batch4, 0)
    assert torch.isfinite(loss2)
    assert torch.isfinite(loss4)
