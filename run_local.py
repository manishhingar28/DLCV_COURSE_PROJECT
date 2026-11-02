"""
Run the whole project on a local machine with an NVIDIA GPU (Windows / Linux / macOS-CPU).
Everything is resumable: re-run the same command after any interruption and finished steps are skipped.

Typical use (features already made on Colab, downloaded from Drive into ./features):
    python run_local.py --features features

From raw videos (downloads questions automatically, needs NExTVideo.zip, ~23 GB):
    python run_local.py --video_zip D:/data/NExTVideo.zip

Quick trial (300 videos, 1 seed):
    python run_local.py --video_zip D:/data/NExTVideo.zip --quick
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONUTF8"] = "1"          # Windows: avoid encoding crashes
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
PY = sys.executable
ANN_URL = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/"
METHODS = ["fixed_topk", "fixed_topk_ras", "ffs", "ffs_ras"]
ABLATIONS = ["ffs_ras_fixedbeta", "ffs_k8", "ffs_ras_k8"]


def run(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\nSTEP FAILED (exit code {r.returncode}). Copy the error above and send it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"), help="where checkpoints/results go")
    ap.add_argument("--features", default=None, help="folder with train.pt + val.pt (skip video processing)")
    ap.add_argument("--video_zip", default=None, help="NExTVideo.zip (only needed to make features / pictures)")
    ap.add_argument("--model", default="EVA02-L-14")
    ap.add_argument("--pretrained", default="merged2b_s4b_b131k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="300 videos, 1 seed: a short trial")
    ap.add_argument("--ablations", action="store_true", help="also train the optional ablations (seed 0)")
    ap.add_argument("--extra", nargs="+", default=[], help="extra configs to train for all seeds, e.g. fixed_topk_nocue ffs_nocue")
    args = ap.parse_args()

    import torch
    gpu = torch.cuda.is_available()
    print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if gpu else "NONE (will be slow!)")
    tag = args.model + ("_quick" if args.quick else "")
    seeds = [args.seeds[0]] if args.quick else args.seeds
    os.makedirs(args.work, exist_ok=True)

    # 1) questions / answers
    ann = os.path.join(args.work, "nextqa_ann")
    os.makedirs(ann, exist_ok=True)
    for f in ["train.csv", "val.csv", "map_vid_vidorID.json"]:
        p = os.path.join(ann, f)
        if not os.path.exists(p):
            print("downloading", f)
            urllib.request.urlretrieve(ANN_URL + f, p)

    # 2) features
    feat = args.features or os.path.join(args.work, f"features_{tag}")
    have = all(os.path.exists(os.path.join(feat, f"{s}.pt")) for s in ["train", "val"])
    if have:
        print("features found in", feat, "-> skipping video processing")
    else:
        if not args.video_zip:
            sys.exit(f"No features in {feat}. Either download the features folder from Drive and pass "
                     f"--features <folder>, or pass --video_zip <path to NExTVideo.zip>.")
        cmd = [PY, "scripts/extract_features.py", "--video_zip", args.video_zip, "--ann_dir", ann,
               "--out_dir", feat, "--n_frames", "32", "--model", args.model, "--pretrained", args.pretrained]
        if args.quick:
            cmd += ["--max_videos", "300"]
        run(cmd)

    # 3) downstream model
    ck = os.path.join(args.work, f"checkpoints_{tag}")
    ds = os.path.join(ck, "downstream", "downstream.pt")
    if os.path.exists(os.path.join(ck, "downstream", "history.json")):
        print("downstream model already trained -> skip")
    else:
        run([PY, "train_downstream.py", "--config", "configs/downstream.yaml", "--data_dir", feat,
             "--out_dir", os.path.join(ck, "downstream")])

    # 4) selectors (finished runs have history.json)
    jobs = [(m, s) for s in seeds for m in METHODS]
    jobs += [(m, s) for s in seeds for m in args.extra]
    if args.ablations:
        jobs += [(m, seeds[0]) for m in ABLATIONS]
    for m, s in jobs:
        if os.path.exists(os.path.join(ck, m, f"seed{s}", "history.json")):
            print(f"skip (done) {m} seed{s}")
            continue
        run([PY, "train_selector.py", "--config", f"configs/{m}.yaml", "--data_dir", feat,
             "--downstream", ds, "--out_dir", os.path.join(ck, m), "--seed", str(s)])

    # 5) evaluation
    res = os.path.join(args.work, f"results_{tag}")
    run([PY, "evaluate.py", "--data_dir", feat, "--downstream", ds, "--runs", ck, "--out_dir", res])
    print("\n" + open(os.path.join(res, "results_table.md"), encoding="utf-8").read())

    # 6) qualitative pictures (needs the videos)
    if args.video_zip and os.path.exists(args.video_zip):
        run([PY, "scripts/visualize.py", "--results", res, "--features", os.path.join(feat, "val.pt"),
             "--ann_dir", ann, "--video_zip", args.video_zip, "--n", "4"])
    else:
        print("(skipping example pictures: pass --video_zip to make them)")

    print(f"\nDONE. Results: {res}\nCheckpoints: {ck}\nSend results_table.md and results.json from {res}.")


if __name__ == "__main__":
    main()
