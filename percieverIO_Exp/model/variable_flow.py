from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from percieverIO_Exp._vendor import ensure_perceiver_vendor_on_path

ensure_perceiver_vendor_on_path()

from perceiver.model.core import (  # noqa: E402
    DecoderConfig,
    EncoderConfig,
    FourierPositionEncoding,
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    QueryProvider,
)


@dataclass
class VariableFlowEncoderConfig(EncoderConfig):
    image_shape: Sequence[int] = (256, 256)
    raw_input_channels: int = 12
    input_width: int = 64
    num_frequency_bands: int = 16


@dataclass
class VariableFlowDecoderConfig(DecoderConfig):
    image_shape: Sequence[int] = (256, 256)
    raw_query_channels: int = 56
    query_width: int = 64
    query_stride: int = 4
    num_frequency_bands: int = 16


@dataclass
class VariableFlowConfig:
    encoder: VariableFlowEncoderConfig = field(default_factory=VariableFlowEncoderConfig)
    decoder: VariableFlowDecoderConfig = field(default_factory=VariableFlowDecoderConfig)
    num_latents: int = 512
    num_latent_channels: int = 256
    activation_checkpointing: bool = False
    activation_offloading: bool = False

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "VariableFlowConfig":
        model_cfg = dict(config)
        image_size = tuple(model_cfg.get("image_size", (256, 256)))
        num_frequency_bands = int(model_cfg.get("num_frequency_bands", 16))
        input_width = int(model_cfg.get("input_width", 64))
        query_width = int(model_cfg.get("query_width", 64))
        raw_input_channels = int(model_cfg.get("raw_input_channels", 12))
        raw_query_channels = int(model_cfg.get("raw_query_channels", 56))
        depth = int(model_cfg.get("depth", 8))
        encoder = VariableFlowEncoderConfig(
            image_shape=image_size,
            raw_input_channels=raw_input_channels,
            input_width=input_width,
            num_frequency_bands=num_frequency_bands,
            num_cross_attention_heads=int(model_cfg.get("cross_attention_heads", 1)),
            num_self_attention_heads=int(model_cfg.get("self_attention_heads", 8)),
            num_self_attention_layers_per_block=depth,
            num_self_attention_blocks=1,
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        decoder = VariableFlowDecoderConfig(
            image_shape=image_size,
            raw_query_channels=raw_query_channels,
            query_width=query_width,
            query_stride=int(model_cfg.get("query_stride", 4)),
            num_frequency_bands=num_frequency_bands,
            num_cross_attention_heads=int(model_cfg.get("cross_attention_heads", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            num_latents=int(model_cfg.get("num_latents", 512)),
            num_latent_channels=int(model_cfg.get("latent_dim", 256)),
            activation_checkpointing=bool(model_cfg.get("activation_checkpointing", False)),
            activation_offloading=bool(model_cfg.get("activation_offloading", False)),
        )


class VariableFlowInputAdapter(InputAdapter):
    def __init__(self, raw_input_channels: int, input_width: int, image_shape: Sequence[int], num_frequency_bands: int):
        position_encoding = FourierPositionEncoding(tuple(image_shape), num_frequency_bands=num_frequency_bands)
        pos_channels = position_encoding.num_position_encoding_channels(include_positions=False)
        super().__init__(input_width + pos_channels)
        self.position_encoding = position_encoding
        self.linear = nn.Linear(raw_input_channels, input_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., 2:4]
        pos = self.position_encoding._position_encodings(xy, include_positions=False)
        return torch.cat([self.linear(x), pos], dim=-1)


class VariableFlowQueryProvider(nn.Module, QueryProvider):
    def __init__(self, raw_query_channels: int, query_width: int, image_shape: Sequence[int], num_frequency_bands: int):
        super().__init__()
        self.position_encoding = FourierPositionEncoding(tuple(image_shape), num_frequency_bands=num_frequency_bands)
        self.linear = nn.Linear(raw_query_channels, query_width)
        self._num_query_channels = query_width + self.position_encoding.num_position_encoding_channels(
            include_positions=False
        )

    @property
    def num_query_channels(self) -> int:
        return self._num_query_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., :2]
        pos = self.position_encoding._position_encodings(xy, include_positions=False)
        return torch.cat([self.linear(x), pos], dim=-1)


class VariableFlowOutputAdapter(OutputAdapter):
    def __init__(self, num_output_query_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GaussianLatentReg(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        flat = latents.reshape(-1, latents.shape[-1])
        mean = flat.mean(dim=0)
        centered = flat - mean
        cov = centered.T @ centered / max(flat.shape[0] - 1, 1)
        ident = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return mean.pow(2).mean() + (cov - ident).pow(2).mean()


class VariableFlowPerceiverIO(nn.Module):
    def __init__(self, config: VariableFlowConfig):
        super().__init__()
        self.config = config
        self.input_adapter = VariableFlowInputAdapter(
            raw_input_channels=int(config.encoder.raw_input_channels),
            input_width=int(config.encoder.input_width),
            image_shape=config.encoder.image_shape,
            num_frequency_bands=int(config.encoder.num_frequency_bands),
        )
        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = self.input_adapter.num_input_channels
        if encoder_kwargs["num_cross_attention_v_channels"] is None:
            encoder_kwargs["num_cross_attention_v_channels"] = self.input_adapter.num_input_channels
        self.encoder = PerceiverEncoder(
            input_adapter=self.input_adapter,
            num_latents=int(config.num_latents),
            num_latent_channels=int(config.num_latent_channels),
            activation_checkpointing=bool(config.activation_checkpointing),
            activation_offloading=bool(config.activation_offloading),
            **encoder_kwargs,
        )
        self.query_provider = VariableFlowQueryProvider(
            raw_query_channels=int(config.decoder.raw_query_channels),
            query_width=int(config.decoder.query_width),
            image_shape=config.decoder.image_shape,
            num_frequency_bands=int(config.decoder.num_frequency_bands),
        )
        self.output_adapter = VariableFlowOutputAdapter(self.query_provider.num_query_channels)
        self.decoder = PerceiverDecoder(
            output_adapter=self.output_adapter,
            output_query_provider=self.query_provider,
            num_latent_channels=int(config.num_latent_channels),
            activation_checkpointing=bool(config.activation_checkpointing),
            activation_offloading=bool(config.activation_offloading),
            **config.decoder.base_kwargs(),
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        query_inputs: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latents = self.encoder(input_tokens, pad_mask=pad_mask)
        query = self.query_provider(query_inputs)
        decoded = self.decoder.cross_attn(query, latents).last_hidden_state
        pred_flow = self.output_adapter(decoded)
        z_content = latents.mean(dim=1)
        return {"pred_flow": pred_flow, "latents": latents, "z_content": z_content}
