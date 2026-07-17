# SETUP — what YOU need to provide / prepare

*Everything the project needs from you (accounts, credentials, environment,
hardware) to be completed. Status legend: ✅ already satisfied · ⚠️ needed soon ·
⏳ needed later. Companion: [PROGRESS.MD](PROGRESS.MD).*

## 1. Local environment (Windows, this machine)

| Item | Status | Detail |
|---|---|---|
| Python | ✅ | 3.14 at `C:\Python314\python.exe` has the full stack (torch 2.10 CPU, PyG 2.8, numpy/pandas/pyarrow/pytest). ⚠️ **PATH quirk:** plain `python` sometimes resolves to Python 3.13 (no packages). Use `C:\Python314\python.exe` or `py -3.14` for all project commands, or reorder PATH. |
| Dependencies | ✅ | `py -3.14 -m pip install -r requirements.txt` re-creates the env. |
| Dataset | ✅ | All 5 collections on disk under `data/npz/tpugraphs/` (7868 files, ≈9 GB). Treated read-only — nothing edits it. |
| Disk headroom | ⚠️ | Reserve **≈7 GiB** for the Phase-3 config shards (`data/cache/config_shards/`), plus a few GB for checkpoints under `artifacts/`. |
| GPU locally | — | None needed; local runs are CPU smoke tests only (<2 min by design). |

## 2. Kaggle account (the critical external dependency)

Needed from **Phase 2 GPU training** onward, and for every leaderboard probe.

1. ⚠️ **Join the competition** and accept its rules at
   <https://www.kaggle.com/competitions/predict-ai-model-runtime> — required both for
   API data access and for **late submissions** (our only scoring mechanism; the
   contest is closed, late submissions still return a public/private LB score).
2. ⚠️ **API token:** Kaggle → Settings → *Create New Token* → save as
   `C:\Users\user\.kaggle\kaggle.json` (used by `kaggle` CLI for submissions and any
   re-download).
3. ⚠️ **Phone-verify** the account if not already — unverified accounts cannot use
   GPU notebooks or submit.
4. ⏳ **GPU quota planning:** Kaggle gives ~30 GPU-hours/week (P100/T4). Phase 2 needs
   ~1–3 h; Phases 3–5 are the heavy consumers (4 layout collections × epochs ×
   seeds/architectures). If quota becomes the bottleneck, a Colab Pro subscription
   (or any CUDA box) is the fallback — the code is path-portable
   (`TPUGRAPHS_DATA_ROOT` env var; outputs to `/kaggle/working` or `artifacts/`).

## 3. Getting the code onto the GPU runtime

Pick one (needed at Phase-2 full training):

- **Recommended:** push this repo to **GitHub** (private is fine) and `git clone` it
  inside the Kaggle/Colab notebook. You'd need to create the repo and give me the
  remote URL — the folder is not a git repository yet (I can `git init` and prepare
  commits on request).
- Alternative: zip `src/ scripts/ configs/ tests/` and upload as a Kaggle *Dataset*
  attached to the notebook (clunkier to iterate).
- On Kaggle, the competition data mounts read-only at
  `/kaggle/input/predict-ai-model-runtime/...` — already handled by
  `src/data/paths.py`; norm stats/inventory regenerate in minutes or can be uploaded.

## 4. Decisions / information only you can give

| # | Question | When | Default if you say nothing |
|---|---|---|---|
| 1 | GitHub repo vs zip-upload for GPU runs (§3) | before P2 full training | I prepare a git repo locally; you create the remote |
| 2 | Kaggle vs Colab as primary GPU | before P2 full training | Kaggle |
| 3 | May I trigger late submissions via your Kaggle API token, or do you upload each `submission.csv` yourself? | end of P2 | **You submit manually** — I generate validated files in `artifacts/submissions/` and tell you when |
| 4 | Course deadline for the final report (sets how long P4/P5 can run) | anytime | Quality-first, no hard date (per your instruction) |
| 5 | Experiment tracking: plain CSV log (current plan) vs Weights & Biases (needs an account/key) | P4 | CSV log |

## 5. Nothing needed for these (already in place)

- Agent team: `.claude/agents/tpu-phase{1..5}-*.md` + `docs/TEAM.md` (the `tpu-phase*`
  agent types auto-register when you start a session in this folder).
- Documentation build chain (`docs/src/build_pdf.py` — headless Chrome + MathJax).
- Data-quality assurance: inventory, tests, EDA are all reproducible
  (`scripts/make_inventory.py`, `pytest -q tests/`, `notebooks/01_eda.ipynb`).

## 6. Quick health check (run anytime)

```bash
cd C:\Users\user\.vscode\.vscode\NTU\SC4000\kaggle---prj
py -3.14 -m pytest -q tests/          # expect: 13 passed
py -3.14 scripts/make_inventory.py    # expect: 7868 files, gate checks OK
```
