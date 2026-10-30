"""
Qualitative figure: frames picked by FFS vs FFS+RAS (ours) on the same question.

    python scripts/visualize.py --results results --features features/val.pt \
        --ann_dir NExT-QA/dataset/nextqa --video_zip /content/NExTVideo.zip --n 4

Picks questions where ours is correct and FFS is wrong (and vice versa for honesty),
and saves results/qualitative_<i>.png.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import index_videos, read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--features", default="features/val.pt")
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--video_zip", default=None)
    ap.add_argument("--video_root", default=None)
    ap.add_argument("--a", default="ffs")
    ap.add_argument("--b", default="ffs_ras")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n_frames", type=int, default=32)
    args = ap.parse_args()

    A = json.load(open(os.path.join(args.results, f"selections_{args.a}.json")))
    B = json.load(open(os.path.join(args.results, f"selections_{args.b}.json")))
    feats = torch.load(args.features, weights_only=False)
    vids = feats["video_ids"]
    rows = {f"{r['video']}_{r['qid']}": r for r in csv.DictReader(open(os.path.join(args.ann_dir, "val.csv")))}
    vmap = json.load(open(os.path.join(args.ann_dir, "map_vid_vidorID.json")))

    if args.video_zip:
        zf = zipfile.ZipFile(args.video_zip)
        vidx = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist() if n.endswith(".mp4")}
        tmpdir = tempfile.mkdtemp()
    else:
        zf, vidx = None, index_videos(args.video_root)

    wins = [(a, b) for a, b in zip(A, B) if b["pred"] == b["answer"] and a["pred"] != a["answer"]]
    losses = [(a, b) for a, b in zip(A, B) if a["pred"] == a["answer"] and b["pred"] != b["answer"]]
    print(f"{args.b} right & {args.a} wrong: {len(wins)} | {args.a} right & {args.b} wrong: {len(losses)}")
    picks = wins[: args.n] + losses[: max(1, args.n // 4)]

    for i, (a, b) in enumerate(picks):
        vid = vids[a["video_row"]]
        name = os.path.basename(vmap.get(vid, vid))
        path = vidx[name]
        if zf is not None:
            path = zf.extract(path, tmpdir)
        frames = read_frames(path, args.n_frames)
        r = rows[a["qid"]]
        opts = [r[f"a{j}"] for j in range(5)]
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.2))
        for ax, (lab, s) in zip(axes, [(args.a, a), (args.b, b)]):
            sel = s["frames"]
            strip = [frames[t] for t in sel]
            ax.imshow(__import__("numpy").concatenate(strip, 1))
            ok = "correct" if s["pred"] == s["answer"] else "wrong"
            ax.set_title(f"{lab}: {len(sel)} frames {sel} -> '{opts[s['pred']]}' ({ok})", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Q: {r['question']}?   (answer: {opts[int(r['answer'])]})", fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.results, f"qualitative_{i}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
