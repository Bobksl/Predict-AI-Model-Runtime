import os, glob, json
import numpy as np
import pandas as pd

def summarize_npz(path: str, kind: str) -> dict:
    d = np.load(path)
    node_feat = d["node_feat"]
    edge_index = d["edge_index"]
    out = {
        "path": path,
        "bytes": os.path.getsize(path),
        "n_nodes": int(node_feat.shape[0]),
        "n_node_feat": int(node_feat.shape[1]),
        "n_edges": int(edge_index.shape[0]),
    }
    if kind == "tile":
        out["n_configs"] = int(d["config_feat"].shape[0])
        rt = d["config_runtime"].astype(np.float64)
        rn = d["config_runtime_normalizers"].astype(np.float64)
        ratio = rt / np.maximum(rn, 1e-12)
        out["ratio_mean"] = float(np.mean(ratio))
        out["ratio_p50"] = float(np.quantile(ratio, 0.50))
        out["ratio_p10"] = float(np.quantile(ratio, 0.10))
        out["ratio_p90"] = float(np.quantile(ratio, 0.90))
        out["ratio_cv"] = float(np.std(ratio) / (np.mean(ratio) + 1e-12))
    else:
        out["n_configs"] = int(d["config_runtime"].shape[0])
        rt = d["config_runtime"].astype(np.float64)
        out["rt_mean"] = float(np.mean(rt))
        out["rt_p50"] = float(np.quantile(rt, 0.50))
        out["rt_p10"] = float(np.quantile(rt, 0.10))
        out["rt_p90"] = float(np.quantile(rt, 0.90))
        out["rt_cv"] = float(np.std(rt) / (np.mean(rt) + 1e-12))

    out["has_nan"] = bool(np.isnan(node_feat).any())
    out["has_inf"] = bool(np.isinf(node_feat).any())
    return out

def build_manifest(npz_glob: str, kind: str, out_csv: str):
    rows = []
    for p in glob.glob(npz_glob):
        rows.append(summarize_npz(p, kind))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
