"""
Extract frozen CLIP features for NExT-QA (run on a GPU, e.g. Colab).

    python scripts/extract_features.py \
        --video_zip  /content/NExTVideo.zip \
        --ann_dir   NExT-QA/dataset/nextqa \
        --out_dir   features \
        --n_frames  32

For every video: sample n_frames evenly spaced frames -> CLIP ViT-B/32 image features.
For every question: CLIP text features of the question, each option, and "question? option".
Writes features/train.pt and features/val.pt (format documented in src/data.py).

Per-video features are cached in out_dir/cache/ so the script can resume after a
Colab disconnect. Use --max_videos to make a quick small run first.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def read_frames(path, n):
    """Return n evenly spaced RGB frames (uint8 HxWx3)."""
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=2)
        L = len(vr)
        idx = np.linspace(0, L - 1, n * 2 + 1)[1::2].round().astype(int)
        return vr.get_batch(idx.tolist()).asnumpy()
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(path)
        L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, max(L - 1, 0), n * 2 + 1)[1::2].round().astype(int)
        frames, last = [], None
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                last = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(last)
        cap.release()
        frames = [f if f is not None else np.zeros((224, 224, 3), np.uint8) for f in frames]
        return np.stack(frames)


def index_videos(video_root):
    idx = {}
    for p in glob.glob(os.path.join(video_root, "**", "*.mp4"), recursive=True):
        idx[os.path.splitext(os.path.basename(p))[0]] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default=None, help="folder containing the (unzipped) .mp4 files")
    ap.add_argument("--video_zip", default=None, help="OR the NExTVideo.zip itself (read without unzipping)")
    ap.add_argument("--ann_dir", required=True, help="folder with train.csv, val.csv, map_vid_vidorID.json")
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--max_videos", type=int, default=None, help="per split, for a quick test run")
    ap.add_argument("--batch_text", type=int, default=512)
    args = ap.parse_args()

    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer(args.model)
    from PIL import Image

    vmap = json.load(open(os.path.join(args.ann_dir, "map_vid_vidorID.json"), encoding="utf-8"))
    zf = None
    if args.video_zip:
        import tempfile
        import zipfile
        zf = zipfile.ZipFile(args.video_zip)
        vidx = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist() if n.endswith(".mp4")}
        tmpdir = tempfile.mkdtemp()
        print(f"found {len(vidx)} mp4 files inside {args.video_zip}")
    else:
        assert args.video_root, "give --video_root or --video_zip"
        vidx = index_videos(args.video_root)
        print(f"found {len(vidx)} mp4 files under {args.video_root}")
    cache = os.path.join(args.out_dir, "cache")
    os.makedirs(cache, exist_ok=True)

    @torch.no_grad()
    def encode_text(texts):
        out = []
        for s in range(0, len(texts), args.batch_text):
            t = tok(texts[s:s + args.batch_text]).to(dev)
            out.append(F.normalize(model.encode_text(t).float(), dim=-1).cpu())
        return torch.cat(out)

    for split in args.splits:
        rows = list(csv.DictReader(open(os.path.join(args.ann_dir, f"{split}.csv"), encoding="utf-8")))
        vids = sorted({r["video"] for r in rows})
        if args.max_videos:
            vids = vids[: args.max_videos]
        keep = set()
        feats = []
        for v in tqdm(vids, desc=f"{split} videos"):
            cpath = os.path.join(cache, f"{v}.npy")
            if os.path.exists(cpath):
                feats.append(np.load(cpath)); keep.add(v); continue
            stem = os.path.basename(vmap.get(v, v))
            path = vidx.get(stem) or vidx.get(v)
            if path is None:
                print("missing video", v); continue
            try:
                if zf is not None:                       # extract one video, read it, delete it
                    tmp = zf.extract(path, tmpdir)
                    frames = read_frames(tmp, args.n_frames)
                    os.remove(tmp)
                else:
                    frames = read_frames(path, args.n_frames)
            except Exception as e:
                print("failed", v, e); continue
            x = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(dev)
            with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
                f = F.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy().astype(np.float16)
            np.save(cpath, f)
            feats.append(f); keep.add(v)
        video_ids = [v for v in vids if v in keep]
        vrow = {v: i for i, v in enumerate(video_ids)}
        rows = [r for r in rows if r["video"] in vrow]
        print(f"{split}: {len(video_ids)} videos, {len(rows)} questions")

        q = [r["question"].strip() + "?" for r in rows]
        opts = [[r[f"a{j}"].strip() for j in range(5)] for r in rows]
        qa = [[f"{qq} {o}" for o in oo] for qq, oo in zip(q, opts)]
        D = None
        qf = encode_text(q); D = qf.shape[1]
        of = encode_text([o for oo in opts for o in oo]).view(len(rows), 5, D)
        qaf = encode_text([s for ss in qa for s in ss]).view(len(rows), 5, D)
        out = {
            "video_ids": video_ids,
            "video_feats": torch.from_numpy(np.stack(feats)).half(),
            "vid_idx": torch.tensor([vrow[r["video"]] for r in rows]),
            "q": qf.half(), "opt": of.half(), "qa": qaf.half(),
            "answer": torch.tensor([int(r["answer"]) for r in rows]),
            "qtype": [r["type"] for r in rows],
            "qid": [f"{r['video']}_{r['qid']}" for r in rows],
            "meta": {"model": args.model, "pretrained": args.pretrained, "n_frames": args.n_frames},
        }
        torch.save(out, os.path.join(args.out_dir, f"{split}.pt"))
        print("saved", os.path.join(args.out_dir, f"{split}.pt"))


if __name__ == "__main__":
    main()
