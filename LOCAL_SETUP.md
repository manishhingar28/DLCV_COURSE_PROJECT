# Running on a local laptop with an NVIDIA GPU

## 1. One-time setup (≈15 min)

Install **Miniconda** (https://docs.conda.io/en/latest/miniconda.html) if you don't have it. Then, in a terminal
(Windows: open **Anaconda Prompt**):

```bash
git clone https://github.com/manishhingar28/DLCV_COURSE_PROJECT.git
cd DLCV_COURSE_PROJECT
conda create -n ffs python=3.11 -y
conda activate ffs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # works with old drivers too
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and your GPU name. If it prints `False`, update the NVIDIA driver
(https://www.nvidia.com/Download/index.aspx) and try again.

## 2. Get the data

**Option A, fastest (if the full feature extraction finished on Colab).**
In Google Drive open `dlcv_ffs/features_EVA02-L-14/`, download **`train.pt` and `val.pt`** (not the `cache` folder),
and put them in a folder `features` inside the project folder.

**Option B, from the videos.** Download `NExTVideo.zip` (~23 GB) from Google Drive (your `NExTVideo (1).zip` copy) to
the laptop. Video processing then takes ~2–4 h on a laptop GPU.

## 3. Run everything (one command, resumable)

Option A:
```bash
python run_local.py --features features
```
Option B:
```bash
python run_local.py --video_zip "C:/path/to/NExTVideo.zip"
```
Add `--video_zip ...` to option A as well if you have the zip and want the example pictures.

* It trains the answer model, then 4 selectors × 3 seeds, then evaluates. About 1.5–3 h on a laptop GPU.
* If it stops (laptop sleeps, you close it), run **the same command again**. Finished steps are skipped.
* Keep the laptop plugged in and set it not to sleep.
* Outputs go to `work/`: `work/results_EVA02-L-14/results_table.md` is the table to send.

Short trial first (optional): add `--quick` (needs `--video_zip`).
