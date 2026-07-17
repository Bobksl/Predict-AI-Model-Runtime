import numpy as np

p = "npz/tile/xla/train/<SOME_FILE>.npz"
d = dict(np.load(p))

print("Before:")
print("node_feat:", d["node_feat"].dtype, d["node_feat"].shape, "nan?", np.isnan(d["node_feat"]).any())
print("config_runtime_normalizers min:", d["config_runtime_normalizers"].min())

# After: safe ratio target
rt = d["config_runtime"].astype(np.float64)
rn = d["config_runtime_normalizers"].astype(np.float64)
rn_safe = np.where(rn <= 0, np.nan, rn)
ratio = rt / rn_safe

print("\nAfter:")
print("ratio nan count:", np.isnan(ratio).sum(), "ratio finite count:", np.isfinite(ratio).sum())