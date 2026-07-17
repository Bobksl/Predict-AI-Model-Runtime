"""TileRanker shape + parameter-budget checks (brief T4/T8)."""
import torch

from src.data import TpugraphsDataset, make_loader
from src.data.cache import CompactCache
from src.models import build_model, TileRanker


def _cfg():
    return {"type": "tile_sage", "opcode_emb_dim": 32, "hidden_dim": 64,
            "n_layers": 3, "config_proj_dim": 32, "dropout": 0.1}


def test_param_budget_under_500k():
    model = build_model(_cfg())
    n = sum(p.numel() for p in model.parameters())
    assert n < 500_000, f"param count {n} exceeds budget"


def test_forward_shapes_on_smoke_batch(data_root, tmp_path):
    cache = CompactCache(tmp_path / "cache", enabled=True)
    ds = TpugraphsDataset("tile:xla", "train", data_root=str(data_root),
                          k_configs=6, cache=cache, add_reverse_edges=True, seed=0)
    ds.files = ds.files[:4]
    loader = make_loader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))

    model: TileRanker = build_model(_cfg())
    scores = model(batch)
    assert scores.shape == (2, 6)
    assert torch.isfinite(scores).all()
