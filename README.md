# FFS-RAS: Redundancy-Aware Flexible Frame Selection for Video QA

**Team:** `TeamXX` — *Name 1 (Roll), Name 2 (Roll), Name 3 (Roll), Name 4 (Roll)*
**Course:** Deep Learning for Computer Vision, 2026

## 1. Project overview

**Assigned paper:** S. Buch, A. Nagrani, A. Arnab, C. Schmid, *Flexible Frame Selection for Efficient Video Reasoning*, CVPR 2025
(CVPR 2025 open-access proceedings, pp. 29071–29082; [project page](https://ffs-video.github.io)).
We found no public code for the paper, so the FFS baseline here is **our own re-implementation** from the paper text.

**What FFS does.** A small selector looks at 32 cheap CLIP frame features plus the question, and passes only the useful frames
to a large video-QA model. Learnable **[CTRL]** tokens compete with the frames inside a differentiable top-*k*, so the number of selected frames
adapts per question (anywhere in `[k − c, k]`).

**Our novelty: Redundancy-Aware Selection (RAS).** FFS scores every frame *independently*, so it often spends its small budget on
near-duplicate neighbouring frames. RAS turns FFS's top-*k* into a **greedy, redundancy-penalised selection**. After each pick,
every remaining frame's score is lowered by `β · similarity(frame, picked frame)`. The penalty strength β is **learned end-to-end**.
[CTRL] tokens are never penalised, so once the remaining frames are mostly repeats, [CTRL] wins the remaining slots and the
selector **stops early**. With β = 0, RAS is *exactly* FFS (it strictly generalises the baseline).

| | Uniform | Fixed top-k | FFS (paper) | **FFS + RAS (ours)** |
|---|---|---|---|---|
| Question-aware | ✗ | ✓ | ✓ | ✓ |
| Adaptive number of frames | ✗ | ✗ | ✓ (relevance only) | ✓ (relevance **and novelty**) |
| Avoids duplicate frames | partly | ✗ | ✗ | ✓ |

Where the modification lives in the code:

| File | What |
|---|---|
| `src/selector.py` → `frame_similarity`, `sequential_topk` (`beta>0` branch), `FrameSelector.beta` | **Our novelty (RAS)** |
| `src/selector.py` → `FrameSelector` with `beta=0` | FFS baseline (encoder, [CTRL] tokens, Gumbel top-k + straight-through) |
| `src/selector.py` → `oracle_count`, `count_hinge_loss`, `greedy_frame_scores`, decoupling in `FrameSelector.forward` | Training recipe for *all* flexible selectors (see §7) |
| `src/downstream.py` | Frozen downstream QA model (stand-in for the paper's 8B VLM) |

## 2. Environment and dependencies

Tested on **Google Colab, T4 GPU (16 GB), Python 3.12, PyTorch 2.x + CUDA 12**. CPU works for the tests and dummy run.

```bash
git clone <this-repo-url> ffs-diverse && cd ffs-diverse
pip install -r requirements.txt
python tests/test_selector.py        # 5 unit tests of the selection operators
```

Easiest route: open **`notebooks/run_on_colab.ipynb`** in Colab and use *Runtime → Run all*. It does every step below.

## 3. Dataset preparation (NExT-QA, multiple choice)

```bash
# 1) questions/answers (official repo)
mkdir -p nextqa_ann && cd nextqa_ann
for f in train.csv val.csv map_vid_vidorID.json; do
  wget https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/$f; done
cd ..
# 2) raw videos (official Google Drive zip)
gdown 1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH -O NExTVideo.zip
# 3) frozen CLIP ViT-B/32 features: 32 evenly spaced frames/video + text features
python scripts/extract_features.py --video_zip NExTVideo.zip --ann_dir nextqa_ann --out_dir features --n_frames 32
```

Expected layout:
```
features/
  train.pt   # 3,870 videos, 34,132 questions
  val.pt     #   570 videos,  4,996 questions  (all results are reported on this split)
  cache/     # per-video .npy (lets extraction resume)
```
`train.pt`/`val.pt` format is documented at the top of `src/data.py`. 10% of *training videos* are held out (`split_seed: 0`)
for checkpoint selection, so the val set is never used for model selection.

## 4. Training

```bash
# Stage 1: downstream QA model, trained on random 1–8 frame subsets, then frozen
python train_downstream.py --config configs/downstream.yaml --data_dir features --out_dir checkpoints/downstream

# Stage 2: selectors trained against the frozen downstream model (seeds 0,1,2)
for s in 0 1 2; do
  python train_selector.py --config configs/fixed_topk.yaml --seed $s   # ablation: always 4 frames
  python train_selector.py --config configs/ffs.yaml        --seed $s   # baseline: FFS, 1–4 frames
  python train_selector.py --config configs/ffs_ras.yaml    --seed $s   # OURS:     FFS + RAS, 1–4 frames
done
# optional ablations
python train_selector.py --config configs/ffs_ras_fixedbeta.yaml   # beta fixed (not learned)
python train_selector.py --config configs/ffs_k8.yaml              # FFS 1–8 frames
python train_selector.py --config configs/ffs_ras_k8.yaml          # ours 1–8 frames
```
Or `bash scripts/run_all.sh features checkpoints results 0 1 2`.

Main hyper-parameters (all in `configs/*.yaml`):

| | Downstream | Selectors |
|---|---|---|
| Model | 2-layer transformer, d=256 + contrastive branch | 2-layer transformer, d=256 |
| Optimiser | AdamW, lr 3e-4, wd 0.05, one-cycle | AdamW, lr 3e-4, wd 0.05, one-cycle |
| Epochs / batch | 15 / 128 | 10 / 128 |
| Selection | – | k=4, c=3 [CTRL] → 1–4 frames, Gumbel τ=1 |
| RAS | – | β init 1.0 (learned), centred similarity |
| Aux loss | – | λ_count = 1.0 (identical for FFS and ours) |

## 5. Inference and evaluation

```bash
python evaluate.py --data_dir features --downstream checkpoints/downstream/downstream.pt \
                   --runs checkpoints --out_dir results
python scripts/visualize.py --results results --features features/val.pt --ann_dir nextqa_ann \
                   --video_zip NExTVideo.zip --n 4          # qualitative figure
```
This evaluates uniform@{1,2,4,8,16,32}, random@4 and every trained selector found under `checkpoints/<method>/seed*/`,
all with the **same frozen downstream model**. Outputs:

* `results/results_table.md`, `results/results_table.tex`: accuracy (overall / Causal / Temporal / Descriptive), average frames,
  redundancy (mean pairwise cosine of selected frames), relative downstream cost (frames / 32); mean ± std over seeds
* `results/acc_vs_frames.png`: accuracy vs. frames trade-off; `results/frame_hist.png`: how many frames each method picks
* `results/selections_<method>.json`: per-question picks; `results/qualitative_*.png`

## 6. Checkpoints and results

| Item | Link |
|---|---|
| All checkpoints + configs + training logs (`submission_artifacts.zip`) | *add Google Drive link* |
| Results table | `results/results_table.md` *(copy in after running)* |

**Results on NExT-QA val** *(fill in from `results/results_table.md`)*

| Method | Avg frames | Acc | Causal | Temporal | Descr. | Redund.↓ |
|---|---|---|---|---|---|---|
| uniform@4 | 4 | | | | | |
| uniform@32 | 32 | | | | | |
| fixed top-4 | 4 | | | | | |
| FFS (re-impl.) | | | | | | |
| **FFS + RAS (ours)** | | | | | | |

## 7. Differences from the paper (and why)

* **Downstream model.** The paper uses an 8B EVA-CLIP based VLM. We use a small transformer + contrastive (CLIP-similarity) answerer
  over frozen CLIP ViT-B/32 features (~2M params), trained once and then frozen. Absolute accuracy is therefore lower than in the paper.
  All methods share the same downstream model, so the comparison is fair.
* **Auxiliary supervision.** The paper says it supervises the selection "with a set of fixed-k policies", with details in a supplement we
  don't have. Our version: for each question, k\* = the smallest j ∈ [k_low, k_high] for which the frozen model answers correctly from the
  selector's own top-j frames. A margin hinge then makes [CTRL] act as a threshold that admits k\* frames (`count_hinge_loss`).
* **Gradient decoupling.** Without it, the task loss always pushed the selector to use all k frames. So the task gradient only
  flows through a relaxation over frames (which frames). The [CTRL] threshold (how many) is learned by the count loss. Forward
  selections are unchanged.
* These three choices are applied **identically** to FFS and FFS+RAS. The only difference between the two configs is `beta`.

## 8. Code organisation

```
configs/            one YAML per method (+ configs/smoke/ for the CPU dummy run)
src/selector.py     FFS + RAS selection operators, [CTRL], losses          <- novelty here
src/downstream.py   frozen downstream QA model, uniform/random masks
src/data.py         feature-file loader, dev split by video
train_downstream.py stage 1
train_selector.py   stage 2
evaluate.py         all methods on val, tables, plots
scripts/            extract_features.py, visualize.py, run_all.sh, smoke_test.sh, build_notebook.py, make_dummy_data.py
notebooks/          run_on_colab.ipynb (end-to-end)
tests/              unit tests (selection operators, feature extraction)
report/             one-page report LaTeX content
```

Sanity check without data or GPU (~10 min, synthetic features, **not** for reporting): `bash scripts/smoke_test.sh`.

## References
* Buch et al., Flexible Frame Selection for Efficient Video Reasoning, CVPR 2025.
* Xiao et al., NExT-QA, CVPR 2021. Radford et al., CLIP, ICML 2021.
* Xie & Ermon, Reparameterizable Subset Sampling via Continuous Relaxations, IJCAI 2019 (relaxed top-k).
* Carbonell & Goldstein, MMR, SIGIR 1998. Kulesza & Taskar, Determinantal Point Processes for ML, 2012.
