"""Builds notebooks/run_on_colab.ipynb (kept as code so the notebook stays in sync)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# FFS + Redundancy-Aware Selection — run everything on Colab

**Before you start:** Runtime → Change runtime type → **T4 GPU** (free tier works; only small outputs go to your Drive).
Then run the cells top to bottom (Runtime → Run all).

Everything except the raw videos is saved to your Google Drive (`MyDrive/dlcv_ffs/`), so if Colab disconnects you can
re-run and it will **skip what is already done**. Total ~4-6 h of GPU time: on free Colab this may take 2 sessions.

| Step | What it does | Rough time (T4) |
|---|---|---|
| 1–3 | GPU check, Drive, code + packages | 2 min |
| 4 | NExT-QA questions (from GitHub) | <1 min |
| 5 | NExT-QA videos (Google Drive zip) | 10–30 min |
| 6 | EVA02-CLIP-L/14 features for 32 frames/video (resumable) | 1.5–3 h |
| 7 | Train downstream QA model | ~10 min |
| 8 | Train selectors: top-4, top-4+ours, FFS, FFS+ours, 3 seeds (use `SEEDS=[0]` if short on GPU time) | 2–3 h |
| 9 | Evaluate on NExT-QA val, tables + plots | ~2 min |
| 10 | Qualitative figure | ~2 min |
""")

md("## 1. Check the GPU")
code("!nvidia-smi")

md("## 2. Mount Google Drive (all outputs are saved here)")
code("""
from google.colab import drive
drive.mount('/content/drive')
WORK = '/content/drive/MyDrive/dlcv_ffs'
import os; os.makedirs(WORK, exist_ok=True)
print('outputs go to', WORK)
""")

md("""
## 3. Get the code and install packages
Put your team's GitHub URL below. (If the repo is private, upload `ffs-diverse.zip` to
`MyDrive/dlcv_ffs/` instead and leave `REPO_URL` empty.)
""")
code("""
REPO_URL = 'https://github.com/manishhingar28/DLCV_COURSE_PROJECT.git'
%cd /content
if REPO_URL:
    !rm -rf ffs-diverse && git clone -q $REPO_URL ffs-diverse
else:
    !rm -rf ffs-diverse && unzip -q -o $WORK/ffs-diverse.zip -d /content/
%cd /content/ffs-diverse
!pip -q install open_clip_torch timm gdown opencv-python-headless
!pip -q install decord || echo "decord not available - using OpenCV (fine, a bit slower)"
!python tests/test_selector.py
""")

md("## 4. NExT-QA question files (official repo)")
code("""
ANN = '/content/nextqa_ann'
os.makedirs(ANN, exist_ok=True)
base = 'https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa'
for f in ['train.csv', 'val.csv', 'map_vid_vidorID.json']:
    !wget -q -nc -O $ANN/$f $base/$f
!wc -l $ANN/train.csv $ANN/val.csv
""")

md("""
## 5. NExT-QA videos
Downloads the official `NExTVideo.zip` straight to Colab's local disk (NOT your Drive, so a free 15 GB Drive is enough).
If Colab disconnects, this re-downloads (10-30 min); everything else is kept in Drive.

**If gdown fails** ("Too many users have viewed or downloaded this file"):
open https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view in your browser,
click **Add shortcut to Drive**, put it in `MyDrive/dlcv_ffs/` and rename the shortcut to `NExTVideo.zip`
(a shortcut uses none of your storage), then re-run this cell.
""")
code("""
import glob, shutil, subprocess
LOCAL_ZIP = '/content/NExTVideo.zip'
FILE_ID = '1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH'

def ok():
    return os.path.exists(LOCAL_ZIP) and os.path.getsize(LOCAL_ZIP) > 1e9

# 1) a shortcut/copy named NExTVideo.zip anywhere in your Drive
if not ok():
    hits = glob.glob('/content/drive/MyDrive/**/NExTVideo*.zip', recursive=True)
    print('Found in your Drive:', hits)
    for h in hits:
        try:
            print('copying', h, '...'); shutil.copyfile(h, LOCAL_ZIP)
            if ok(): break
        except Exception as e:
            print('  copy failed:', e)

# 2) download through the Google Drive API, logged in as you (often works when gdown is blocked)
if not ok():
    try:
        from google.colab import auth
        auth.authenticate_user()
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        svc = build('drive', 'v3')
        meta = svc.files().get(fileId=FILE_ID, fields='name,size', supportsAllDrives=True).execute()
        print('Drive API sees:', meta['name'], round(int(meta['size'])/1e9, 2), 'GB')
        req = svc.files().get_media(fileId=FILE_ID, supportsAllDrives=True)
        with open(LOCAL_ZIP, 'wb') as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=200 * 1024 * 1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                print(f'  {int(status.progress() * 100)}%', end='', flush=True)
        print()
    except Exception as e:
        print('Drive API download failed:', str(e)[:300])
        if os.path.exists(LOCAL_ZIP) and not ok():
            os.remove(LOCAL_ZIP)

# 3) last resort: gdown
if not ok():
    !gdown $FILE_ID -O $LOCAL_ZIP

!ls -lh $LOCAL_ZIP
assert ok(), 'VIDEO DOWNLOAD FAILED - copy everything printed above and send it'
print('videos OK')
""")

md("""
## 6. Extract frozen features (32 evenly spaced frames per video)
Default: **EVA02-CLIP-L/14** (same EVA-CLIP family as the paper's backbone, much stronger than CLIP B/32).
Takes ~1.5-3 h on a free T4. It is resumable: per-video features are cached in Drive, so after a disconnect just re-run.
* Faster but weaker option: set `FEATURE_MODEL = 'ViT-B-32'`.
* Set `QUICK = True` for a short trial on 300 videos first (numbers will be lower, only to check everything runs).
""")
code("""
QUICK = True     # True = 10-min trial on 300 videos. Change to False for the real run.
FEATURE_MODEL = 'EVA02-L-14'          # or 'ViT-B-32' (faster, weaker)
PRETRAINED = {'EVA02-L-14': 'merged2b_s4b_b131k', 'ViT-B-32': 'openai'}[FEATURE_MODEL]
TAG = FEATURE_MODEL + ('_quick' if QUICK else '')
FEAT = f'{WORK}/features_{TAG}'
extra = '--max_videos 300' if QUICK else ''
!python scripts/extract_features.py --video_zip $LOCAL_ZIP --ann_dir $ANN --out_dir $FEAT --n_frames 32 --model $FEATURE_MODEL --pretrained $PRETRAINED $extra
!ls -lh $FEAT/*.pt
assert os.path.exists(f'{FEAT}/train.pt') and os.path.exists(f'{FEAT}/val.pt'), \
    'FEATURE EXTRACTION FAILED - scroll up in this cell output for the first error'
print('features OK')
""")

md("""
## 7. Train the downstream QA model (stand-in for the big frozen VLM), then freeze it
""")
code("""
CK = f'{WORK}/checkpoints_{TAG}'
if not os.path.exists(f'{CK}/downstream/downstream.pt'):
    !python train_downstream.py --config configs/downstream.yaml --data_dir $FEAT --out_dir $CK/downstream
!tail -3 $CK/downstream/train_log.txt
assert os.path.exists(f'{CK}/downstream/downstream.pt'), 'DOWNSTREAM TRAINING FAILED - see error above'
""")

md("""
## 8. Train the frame selectors
* `fixed_topk` – always 4 frames (no [CTRL])  — ablation
* `fixed_topk_ras` – always 4 frames + our redundancy-aware selection — **ours at a fixed budget**
* `ffs` – the paper's Flexible Frame Selector, 1–4 frames — **baseline**
* `ffs_ras` – FFS + Redundancy-Aware Selection — **ours**

Three seeds each, so the table can show mean ± std. Already-finished runs are skipped.
""")
code("""
SEEDS = [0, 1, 2] if not QUICK else [0]
METHODS = ['fixed_topk', 'fixed_topk_ras', 'ffs', 'ffs_ras']
for s in SEEDS:
    for m in METHODS:
        if os.path.exists(f'{CK}/{m}/seed{s}/selector.pt'):
            print('skip', m, s); continue
        !python train_selector.py --config configs/{m}.yaml --data_dir $FEAT --downstream $CK/downstream/downstream.pt --out_dir $CK/{m} --seed {s}
""")

md("""
### 8b. (Optional) extra ablations for the report
* `ffs_ras_fixedbeta` – our method with β fixed at 1 (is learning β useful?)
* `ffs_k8`, `ffs_ras_k8` – wider range, 1–8 frames
""")
code("""
RUN_ABLATIONS = False
if RUN_ABLATIONS:
    for m in ['ffs_ras_fixedbeta', 'ffs_k8', 'ffs_ras_k8']:
        if not os.path.exists(f'{CK}/{m}/seed0/selector.pt'):
            !python train_selector.py --config configs/{m}.yaml --data_dir $FEAT --downstream $CK/downstream/downstream.pt --out_dir $CK/{m} --seed 0
""")

md("## 9. Evaluate everything on the NExT-QA validation set (4,996 questions)")
code("""
RES = f'{WORK}/results_{TAG}'
!python evaluate.py --data_dir $FEAT --downstream $CK/downstream/downstream.pt --runs $CK --out_dir $RES
from IPython.display import Markdown, Image, display
display(Markdown(open(f'{RES}/results_table.md').read()))
display(Image(f'{RES}/acc_vs_frames.png', width=520))
display(Image(f'{RES}/frame_hist.png', width=520))
""")

md("## 10. Qualitative examples (FFS vs ours on the same question)")
code("""
!python scripts/visualize.py --results $RES --features $FEAT/val.pt --ann_dir $ANN --video_zip $LOCAL_ZIP --n 4
import glob
for p in sorted(glob.glob(f'{RES}/qualitative_*.png')):
    display(Image(p, width=800))
""")

md("""
## 11. Package checkpoints + logs for submission
Creates `submission_artifacts.zip` in your Drive (checkpoints, configs, logs, results).
Share it with "Anyone with the link" and paste the link into the README.
""")
code("""
!cd $WORK && zip -q -r submission_artifacts.zip checkpoints_$TAG results_$TAG -x "*/cache/*"
!ls -lh $WORK/submission_artifacts.zip
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs("notebooks", exist_ok=True)
for c in nb["cells"]:
    src = c["source"].split("\n")
    c["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]
json.dump(nb, open("notebooks/run_on_colab.ipynb", "w"), indent=1)
print("wrote notebooks/run_on_colab.ipynb with", len(cells), "cells")
