from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from bioseq_ood.config.loader import load_config
from bioseq_ood.data.preprocess_data import encode_dataframe, remove_static_aav_area
from bioseq_ood.models.registry import build_model


def predict_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    sequences: Iterable[str] | None = None,
    input_csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    config = load_config(config_path)
    dataset_name = config["dataset"]["name"]
    if sequences is None:
        if input_csv_path is None:
            raise ValueError("Provide either sequences or input_csv_path.")
        df = pd.read_csv(input_csv_path)
    else:
        df = pd.DataFrame({"sequence": list(sequences)})

    x = encode_dataframe(df, dataset_name)
    if dataset_name == "aav":
        x = remove_static_aav_area(x)
    vocab_size = int(np.max(x)) + 1 if x.size else 0
    seq_len = int(x.shape[1])
    model = build_model(config.get("model_name", "cnn"), config["model"], vocab_size=vocab_size, seq_len=seq_len)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        ckpt_state = state["state_dict"]
        # Mean Teacher checkpoints contain both student (model.*) and teacher
        # (teacher_model.*). Prefer the teacher for inference when available.
        teacher_state = {
            k.replace("teacher_model.", "", 1): v
            for k, v in ckpt_state.items()
            if k.startswith("teacher_model.")
        }
        if teacher_state:
            state = teacher_state
        else:
            state = {k.replace("model.", "", 1): v for k, v in ckpt_state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(x, dtype=torch.long, device=device)).detach().cpu().numpy().reshape(-1)

    result = df.copy()
    result["prediction"] = preds
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result
