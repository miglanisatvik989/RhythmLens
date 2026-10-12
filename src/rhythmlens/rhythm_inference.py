"""Baseline logits from the exact training transform; no quality or calibration."""

from __future__ import annotations

import numpy as np
import torch

from rhythmlens.preprocessing import PreprocessingState, transform_physical
from rhythmlens.rhythm_model import RhythmCNN


def predict_physical(
    model: RhythmCNN,
    state: PreprocessingState,
    physical: np.ndarray,
    *,
    sampling_rate_hz: float,
    units: str,
) -> np.ndarray:
    """Return uncalibrated four-class logits in N,A,O,~ order.

    Call with a model already in evaluation mode; this avoids changing shared
    model state in a future inference service.
    """
    if model.training:
        raise ValueError("Inference requires model.eval()")
    processed = transform_physical(
        physical, state, sampling_rate_hz=sampling_rate_hz, units=units
    )
    device = next(model.parameters()).device
    signal = torch.from_numpy(processed)[None, None, :].to(device)
    mask = torch.ones(1, processed.size, dtype=torch.bool, device=device)
    with torch.inference_mode():
        return model(signal, mask)[0].cpu().numpy().copy()
