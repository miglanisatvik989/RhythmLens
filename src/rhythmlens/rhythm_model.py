"""Compact whole-record CNN whose logits do not depend on padding length."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    stem_channels: int = 32
    stem_kernel: int = 15
    channels: tuple[int, ...] = (32, 64, 128)
    kernel: int = 7
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if len(self.channels) != 3 or any(
            type(c) is not int or c < 2 for c in (self.stem_channels, *self.channels)
        ):
            raise ValueError(
                "Expected three stages and at least two channels per layer"
            )
        if any(
            type(k) is not int or k <= 0 or k % 2 != 1
            for k in (self.stem_kernel, self.kernel)
        ):
            raise ValueError("Convolution kernels must be positive and odd")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Dropout must be in [0,1)")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["channels"] = list(self.channels)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> ModelConfig:
        value = dict(value)
        if "channels" in value:
            value["channels"] = tuple(value["channels"])
        return cls(**value)


class ChannelNorm(nn.Module):
    """Normalize channels independently at each sample, excluding the time axis."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # where, rather than multiplication, also discards nonfinite masked values.
    return torch.where(
        mask[:, None, :], x, torch.zeros((), dtype=x.dtype, device=x.device)
    )


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1)
    if (count == 0).any():
        raise ValueError("Cannot pool a recording with no valid samples")
    return apply_mask(x, mask).sum(dim=-1) / count[:, None].to(x.dtype)


class MaskedResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, kernel: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(
            incoming, outgoing, kernel, stride=stride, padding=kernel // 2
        )
        self.norm1 = ChannelNorm(outgoing)
        self.conv2 = nn.Conv1d(outgoing, outgoing, kernel, padding=kernel // 2)
        self.norm2 = ChannelNorm(outgoing)
        self.skip = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Conv1d(incoming, outgoing, 1, stride=stride)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = apply_mask(x, mask)
        # Odd kernels with symmetric padding sample centers at 0,stride,... .
        # A prefix of L samples therefore becomes ceil(L/stride) valid positions.
        down = mask[:, :: self.stride]
        residual = apply_mask(self.skip(x), down)
        x = apply_mask(torch.relu(self.norm1(self.conv1(x))), down)
        x = apply_mask(self.norm2(self.conv2(x)), down)
        return apply_mask(torch.relu(x + residual), down), down


class RhythmCNN(nn.Module):
    model_version = "rhythm-cnn-v1"

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.stem = nn.Conv1d(
            1,
            self.config.stem_channels,
            self.config.stem_kernel,
            padding=self.config.stem_kernel // 2,
        )
        self.stem_norm = ChannelNorm(self.config.stem_channels)
        blocks, incoming = [], self.config.stem_channels
        for outgoing, stride in zip(self.config.channels, (1, 2, 2), strict=True):
            blocks.append(
                MaskedResidualBlock(incoming, outgoing, self.config.kernel, stride)
            )
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(self.config.dropout)
        self.head = nn.Linear(incoming, 4)

    def features(
        self, signal: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            signal.ndim != 3
            or signal.shape[1] != 1
            or signal.shape[0] == 0
            or signal.shape[-1] == 0
        ):
            raise ValueError("Expected a nonempty [batch,1,time] input")
        if (
            not signal.is_floating_point()
            or mask.dtype != torch.bool
            or mask.shape != (signal.shape[0], signal.shape[-1])
        ):
            raise ValueError("Expected a boolean [batch,time] valid-sample mask")
        if mask.device != signal.device:
            raise ValueError("Signal and mask must be on the same device")
        lengths = mask.sum(dim=1)
        expected = (
            torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
        )
        if (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError("Valid samples must form a nonempty prefix")
        if not torch.isfinite(signal.masked_select(mask[:, None, :])).all():
            raise ValueError("Nonfinite valid signal samples")
        x = apply_mask(signal, mask)
        x = apply_mask(torch.relu(self.stem_norm(self.stem(x))), mask)
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, mask = self.features(signal, mask)
        return self.head(self.dropout(masked_average(features, mask)))


class TemporalCNN(RhythmCNN):
    """Native-rate dilated candidate; masked after every convolution."""

    model_version = "rhythm-dilated-v1"
    dilations = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, config=None):
        super().__init__(config)
        blocks, incoming = [], self.config.stem_channels
        for index, dilation in enumerate(self.dilations):
            outgoing = self.config.channels[min(index, 2)]
            block = MaskedResidualBlock(incoming, outgoing, self.config.kernel, 1)
            for conv in (block.conv1, block.conv2):
                conv.dilation = (dilation,)
                conv.padding = (dilation * (self.config.kernel // 2),)
            blocks.append(block)
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)


def build_model(version, config):
    models = {"rhythm-cnn-v1": RhythmCNN, "rhythm-dilated-v1": TemporalCNN}
    if version not in models:
        raise ValueError(f"Unsupported model version: {version}")
    return models[version](ModelConfig.from_dict(config))


def temporal_capability(model):
    config = model.config
    if model.model_version == "rhythm-dilated-v1":
        receptive_field = config.stem_kernel + 2 * (config.kernel - 1) * sum(
            model.dilations
        )
    else:
        receptive_field, jump = config.stem_kernel, 1
        for stride in (1, 2, 2):
            receptive_field += (config.kernel - 1) * jump
            jump *= stride
            receptive_field += (config.kernel - 1) * jump
    return {
        "model_version": model.model_version,
        "receptive_field_samples": receptive_field,
        "receptive_field_seconds": receptive_field / 300,
        "pooling": "masked global average",
        "sampling_rate_hz": 300,
    }
