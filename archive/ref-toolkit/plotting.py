import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("manifest_tile_xla_train.csv")

plt.figure()
plt.hist(df["n_nodes"], bins=100)
plt.yscale("log")
plt.xscale("log")
plt.title("tile:xla train — n_nodes distribution")
plt.xlabel("n_nodes (log)")
plt.ylabel("count (log)")
plt.tight_layout()
plt.savefig("fig_n_nodes_tile_train.png", dpi=200)