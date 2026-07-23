# Submission Guide — how to submit on Kaggle

This produces the competition `submission.csv` (all 5 collections) from the trained GNN
checkpoints and submits it to
[**Predict AI Model Runtime**](https://www.kaggle.com/competitions/predict-ai-model-runtime/).

The ready-to-run notebook is **`notebooks/kaggle_submission.ipynb`** (it did not exist
before; it does now). Its header repeats these steps; this file is the fuller reference.

---

## What you need first: the 5 checkpoints in one place

Inference needs one `best.pt` per collection:

| Collection | Checkpoint (`<run>_sage/best.pt`) | Where it is trained |
|---|---|---|
| `tile:xla` | `tile_sage_listmle` (or `tile_sage_hinge`) | **locally** (CPU is fine) — `python scripts/train.py --config configs/tile_sage_listmle.yaml` |
| `layout:xla:random` | `layout_xla_random_sage` | **Kaggle GPU** kernel `tpugraphs-layout-training-phase-3` |
| `layout:xla:default` | `layout_xla_default_sage` | same kernel |
| `layout:nlp:random` | `layout_nlp_random_sage` | same kernel |
| `layout:nlp:default` | `layout_nlp_default_sage` | same kernel |

The tile checkpoint already exists locally; the four layout checkpoints come out of the
GPU training kernel's **Output**.

## Step 1 — package the checkpoints as a Kaggle dataset

Collect all five `best.pt` into one folder and upload it once as a private dataset. From the
repo root, after the layout kernel has finished:

```bash
# download the 4 layout checkpoints from the training kernel output
py -3.14 -m kaggle kernels output bobksl/tpugraphs-layout-training-phase-3 -p artifacts/kaggle_ckpts

# stage all 5 into one folder (4 layout from the kernel + the local tile one)
mkdir -p submit_ckpts
cp -r artifacts/kaggle_ckpts/artifacts/checkpoints/layout_*_sage submit_ckpts/
cp -r artifacts/checkpoints/tile_sage_listmle submit_ckpts/

# create + push a Kaggle dataset (edit the id/title once)
py -3.14 -m kaggle datasets init -p submit_ckpts
#   -> edit submit_ckpts/dataset-metadata.json: set "id": "bobksl/tpugraphs-checkpoints"
py -3.14 -m kaggle datasets create -p submit_ckpts
```

(You can skip the dataset entirely and instead attach the training kernel's **Notebook
Output** directly — the notebook searches all of `/kaggle/input`. But that output only has the
4 layout checkpoints, so you would still need the tile one added somewhere.)

## Step 2 — run the submission notebook

1. On Kaggle: **Create → Notebook**, then *File → Import Notebook* → upload
   `notebooks/kaggle_submission.ipynb`.
2. **Add Data** (right panel): add the **competition** `predict-ai-model-runtime` **and** your
   **checkpoints dataset** from Step 1.
3. **Settings:** Internet **ON** (needed to `git clone` the code). GPU optional (inference is
   light; ~50 layout + 844 tile test graphs).
4. **Run All.** The final cell validates every ranking is a valid permutation and writes
   `/kaggle/working/submission.csv`.

## Step 3 — submit

- **From the notebook:** *Save Version* (runs all) → when it finishes, open the version →
  **Submit to Competition** (uses the notebook's `submission.csv` output), **or**
- **CSV upload:** download `submission.csv` from the notebook output and use *Submit
  Predictions* on the competition page.

Late submissions are accepted (the contest is closed) and return public/private LB scores —
exactly what we want to verify the result.

---

## Fully-local alternative (no notebook)

You can also build the CSV on your machine (layout inference is slower on CPU but works),
once all 5 checkpoints are under `artifacts/checkpoints/`:

```bash
# tile
py -3.14 scripts/make_submission.py --checkpoint artifacts/checkpoints/tile_sage_listmle/best.pt \
    --out artifacts/submissions/submission_tile.csv
# each layout collection
py -3.14 scripts/make_layout_submission.py --checkpoint artifacts/checkpoints/layout_xla_random_sage/best.pt
# ... repeat for the other 3 layout checkpoints ...
# combine all 5 into one file
py -3.14 scripts/combine_submission.py --out artifacts/submissions/submission_5col.csv
```

Then upload `submission_5col.csv` via *Submit Predictions*.

---

## Current status (2026-07-19)

- Tile checkpoint: **ready** (`tile_sage_listmle`, valid OPA ≈ 0.878).
- Layout checkpoints: **training on Kaggle GPU** (kernel `tpugraphs-layout-training-phase-3`).
  Once it reports `checkpoint saved` for all four, do Step 1.
- The submission notebook and this guide are ready now; they run as soon as the four layout
  checkpoints exist.
