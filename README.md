# FFS + Similarity Cues: Better Learned Frame Selection for Video QA

**Team:** Mitra Mandal: Abhishek Kumar Jha (230106001), Aditya Dhaniyaal (230106002), Dev Shukla (230106020), Dhruv Mishra (230106022), Manish Hingar (230106040)
**Course:** Deep Learning for Computer Vision, 2026

## 1. Project overview

**Assigned paper:** S. Buch, A. Nagrani, A. Arnab, C. Schmid, *Flexible Frame Selection for Efficient Video Reasoning*, CVPR 2025
(CVPR 2025 open-access proceedings, pp. 29071–29082; [project page](https://ffs-video.github.io)).
We found no public code for the paper, so the FFS baseline here is **our own re-implementation** from the paper text.

**What FFS does.** A small selector looks at 32 cheap CLIP frame features plus the question, and passes only the useful frames
to a large video-QA model. Learnable **[CTRL]** tokens compete with the frames inside a differentiable top-*k*, so the number of selected frames
adapts per question (anywhere in `[k − c, k]`).

**Our novelty: similarity-cue injection into the FFS selector.** FFS has to *learn* which frames matter from the
answer loss alone, through a straight-through top-k. With frozen features this signal is weak: the learned selector
barely beats uniform frames. We give the selector four **zero-shot relevance cues per frame**, computed from the same frozen
EVA02-CLIP features with no labels and no extra model:
cosine to the question, best and second-best "question + option" similarity, and their gap (how clearly the frame separates the answers).
They are standardised within each video, added to the frame tokens, and fed to the selection logit through a linear skip
(`l_t = g(h_t) + w·c_t`). FFS is the special case `w = 0`, so the selector only has to learn a *correction* to a strong zero-shot ranker.

**Ablation: Redundancy-Aware Selection (RAS).** We also tested a greedy top-k that penalises frames similar to already
picked ones (submodular / DPP-style). It did **not** help: see the results and section 7.

Where the modification lives in the code:

| File | What |
|---|---|
| `src/selector.py` → `FrameSelector.frame_cues`, `cue_proj`, `cue_logit` (`sim_feats: true`) | **Our novelty (similarity cues)** |
| `src/selector.py` → `frame_similarity`, `sequential_topk` (`beta>0`), `ras_sweep.py` | RAS ablation |
| `src/selector.py` → `FrameSelector` with `sim_feats: false, beta: 0` | FFS baseline (encoder, [CTRL] tokens, Gumbel top-k + straight-through) |
| `src/selector.py` → `oracle_count`, `count_hinge_loss`, `greedy_frame_scores`, decoupling in `FrameSelector.forward` | Training recipe for *all* flexible selectors (see §7) |
| `src/downstream.py` | Frozen downstream QA model (stand-in for the paper's 8B VLM) |

## 2. Environment and dependencies

Reported results: **Windows, NVIDIA RTX 2060 (6 GB), Python 3.11, PyTorch 2.5.1 + CUDA 11.8**. Features were extracted on
**Google Colab (T4)**. CPU works for the tests and the dummy run.

```bash
git clone <this-repo-url> ffs-diverse && cd ffs-diverse
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU build (pick your CUDA)
pip install -r requirements.txt
python tests/test_selector.py        # 5 unit tests of the selection operators
```

Easiest route: open **`notebooks/run_on_colab.ipynb`** in Colab and use *Runtime → Run all*. It does every step below.
On a local NVIDIA GPU: see **`LOCAL_SETUP.md`** and run `python run_local.py --features features --extra fixed_topk_nocue ffs_nocue`
(one resumable command; this is how the reported results were produced, on an RTX 2060).

## 3. Dataset preparation (NExT-QA, multiple choice)

```bash
# 1) questions/answers (official repo)
mkdir -p nextqa_ann && cd nextqa_ann
for f in train.csv val.csv map_vid_vidorID.json; do
  wget https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/$f; done
cd ..
# 2) raw videos (official Google Drive zip)
gdown 1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH -O NExTVideo.zip
# 3) frozen EVA02-CLIP-L/14 features (same EVA-CLIP family as the paper): 32 evenly spaced frames/video + text
python scripts/extract_features.py --video_zip NExTVideo.zip --ann_dir nextqa_ann --out_dir features --n_frames 32 \
       --model EVA02-L-14 --pretrained merged2b_s4b_b131k      # faster/weaker: --model ViT-B-32 --pretrained openai
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
  python train_selector.py --config configs/fixed_topk.yaml --seed $s   # OURS: top-4 + cues
  python train_selector.py --config configs/fixed_topk_nocue.yaml --seed $s  # baseline: top-4, FFS-style input (no cues)
  python train_selector.py --config configs/fixed_topk_ras.yaml --seed $s   # ablation: top-4 + cues + RAS
  python train_selector.py --config configs/ffs_nocue.yaml  --seed $s   # baseline: FFS re-implementation (no cues)
  python train_selector.py --config configs/ffs.yaml        --seed $s   # OURS: FFS + cues, 1–4 frames
  python train_selector.py --config configs/ffs_ras.yaml    --seed $s   # ablation: FFS + cues + RAS
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
| Model | 2-layer transformer, d=256 + contrastive branch | 2-layer transformer, d=256 + per-frame similarity cues |
| Optimiser | AdamW, lr 1e-4, wd 0.1, dropout 0.3, one-cycle | AdamW, lr 3e-4, wd 0.05, one-cycle |
| Epochs / batch | 20 / 128 | 10 / 128 |
| Selection | – | k=4, c=3 [CTRL] → 1–4 frames, Gumbel τ=1 |
| RAS | – | β init 1.0 (learned), centred similarity |
| Aux loss | – | λ_count = 1.0, count tolerance ε = 0.1 (identical for FFS and ours) |

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
| All checkpoints + configs + training logs + results (`trained_models_and_results.zip`, 147 MB) | [Google Drive](https://drive.google.com/file/d/1A8pq_dSAi95zVbXmI1TmFEUdA_DV450K/view?usp=sharing) |
| Results table | below, and `results_EVA02-L-14/results_table.md` inside the zip |

**Results on NExT-QA val** (4,996 questions, 32-frame pool, same frozen downstream model for all, mean ± std over 3 seeds):

| Method | Frames | Acc | Causal | Temporal | Descr. | Rel. cost |
|---|---|---|---|---|---|---|
| Uniform@1 | 1 | 45.40 | 44.3 | 42.1 | 55.9 | 0.031 |
| Uniform@4 | 4 | 47.94 | 46.8 | 45.1 | 57.7 | 0.125 |
| Uniform@32 | 32 | 48.40 | 46.9 | 46.3 | 57.9 | 1.000 |
| Top-4, FFS-style input | 4 | 48.66 ± 0.19 | 47.8 | 45.7 | 57.7 | 0.125 |
| **Top-4 + cues (ours)** | 4 | **49.71 ± 0.10** | 48.3 | 47.4 | 59.1 | 0.125 |
| Top-4 + cues + RAS (ablation) | 4 | 49.71 ± 0.10 | 48.3 | 47.3 | 59.3 | 0.125 |
| FFS (re-impl., adaptive 1–4) | 1.0 | 46.35 ± 0.19 | 44.9 | 44.0 | 56.2 | 0.031 |
| **FFS + cues (ours)** | 1.0 | **47.20 ± 0.10** | 45.4 | 45.0 | 57.6 | 0.031 |
| FFS + cues + RAS (ablation) | 1.0 | 47.22 ± 0.15 | 45.6 | 44.7 | 57.9 | 0.031 |

* Cues add **+1.05** (top-4) and **+0.85** (adaptive FFS) points, about 5× the seed std, and help every question type.
* With cues, **4 selected frames beat all 32 uniform frames** (49.71 vs 48.40) at 1/8 of the downstream cost.
* Adaptive selectors settled on 1 frame, because this downstream model gains only 3 points from 1 → 32 uniform frames.
* RAS left the picks unchanged (the penalty is small next to the learned score gaps). Learned picks are *more* mutually
  similar than uniform ones (0.92 vs 0.89 mean cosine) yet more accurate: on NExT-QA, evidence clusters around the key moment.
  `ras_sweep.py` re-applies RAS at inference with a scale-invariant penalty to test it when it does change the picks.

## 7. Differences from the paper (and why)

* **Downstream model.** The paper uses an 8B EVA-CLIP based VLM. We use a small transformer + contrastive (CLIP-similarity) answerer
  (~2M params) over frozen EVA02-CLIP-L/14 features, trained once and then frozen. Absolute accuracy is therefore lower than in the paper.
  All methods share the same downstream model, so the comparison is fair.
* **Selector input features.** The paper feeds the selector cheap CLIP-B/32 features. We pre-compute one EVA02-L/14 feature set
  and share it between selector and downstream model. Efficiency is measured, as in the paper, by frames passed to the downstream model.
* **Auxiliary supervision.** The paper says it supervises the selection "with a set of fixed-k policies", with details in a supplement we
  don't have. Our version: for each question, k\* = the smallest j ∈ [k_low, k_high] for which the frozen model answers correctly from the
  selector's own top-j frames gives an answer loss within ε = 0.1 of the best j (plain "first correct j" collapsed to 1 frame,
  because this answer model can often guess from the text alone). A margin hinge then makes [CTRL] act as a threshold that admits k\* frames (`count_hinge_loss`).
* **Gradient decoupling.** Without it, the task loss always pushed the selector to use all k frames. So the task gradient only
  flows through a relaxation over frames (which frames). The [CTRL] threshold (how many) is learned by the count loss. Forward
  selections are unchanged.
* **Our novelty (similarity cues)** came out of a pilot on 300 training videos: without them the selector's scores stayed
  flat (no better than random). See section 1 and the report for details.
* **Regularised downstream model.** The pilot downstream model memorised its training set after 2 epochs, so it uses dropout 0.3,
  weight decay 0.1 and lr 1e-4.
* All tuning used held-out *training* videos only, never the validation set.
* The training recipe above is applied **identically** to every selector. Paired configs differ in one setting only:
  `*_nocue` vs ours: `sim_feats` (false/true); ours vs `*_ras`: `beta` (0/learned).
* An optional relevance-distillation loss (`lambda_distill`, off by default) is included; it did not help in the pilot.

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
report/             main.tex (one-page report, compile on Overleaf) + main.pdf
```

Sanity check without data or GPU (~10 min, synthetic features, **not** for reporting): `bash scripts/smoke_test.sh`.

## References
* Buch et al., Flexible Frame Selection for Efficient Video Reasoning, CVPR 2025.
* Xiao et al., NExT-QA, CVPR 2021. Radford et al., CLIP, ICML 2021.
* Xie & Ermon, Reparameterizable Subset Sampling via Continuous Relaxations, IJCAI 2019 (relaxed top-k).
* Carbonell & Goldstein, MMR, SIGIR 1998. Kulesza & Taskar, Determinantal Point Processes for ML, 2012.