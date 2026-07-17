"""Model registry: build models from a config dict via ``model.type``."""
from .tile_gnn import TileRanker

_REGISTRY = {"tile_sage": TileRanker}


def build_model(model_cfg: dict):
    typ = model_cfg["type"]
    if typ not in _REGISTRY:
        raise KeyError(f"Unknown model.type {typ!r}; known: {sorted(_REGISTRY)}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "type"}
    return _REGISTRY[typ](**kwargs)


__all__ = ["TileRanker", "build_model"]
