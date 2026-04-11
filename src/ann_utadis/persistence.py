from __future__ import annotations

from pathlib import Path

import torch

from .model import AnnUtadis


def save_ann_utadis_bundle(
    model: AnnUtadis,
    output_path: Path,
    metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_criteria": model.num_criteria,
            "num_hidden_components": model.num_hidden_components,
        },
        "metadata": metadata,
    }
    torch.save(bundle, output_path)


def load_ann_utadis_bundle(path: Path, map_location: str = "cpu") -> tuple[AnnUtadis, dict]:
    bundle = torch.load(path, map_location=map_location)
    model_cfg = bundle["model_config"]
    model = AnnUtadis(**model_cfg)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle["metadata"]
