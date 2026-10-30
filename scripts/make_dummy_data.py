"""
Synthetic features in the exact NExT-QA feature format, for testing the pipeline
without videos or a GPU. NOT for reporting results.

Each video = a few long, highly redundant 'scenes' plus 1-3 short 'events'.
The correct answer is tied to the event frames, so a good selector must find
them; uniform sampling with few frames often misses them, and redundant scene
frames waste budget.
"""
import argparse
import os

import torch
import torch.nn.functional as F


def make_split(n_videos, q_per_video, T, D, g):
    feats, vid_idx, q, opt, qa, ans, qtype, qid = [], [], [], [], [], [], [], []
    types = ["CW", "CH", "TN", "TC", "TP", "DO", "DL", "DC"]
    for v in range(n_videos):
        n_scenes = int(torch.randint(2, 5, (1,), generator=g))
        scene_vecs = F.normalize(torch.randn(n_scenes, D, generator=g), dim=-1)
        bounds = torch.sort(torch.randperm(T - 1, generator=g)[: n_scenes - 1] + 1).values.tolist()
        seg = torch.zeros(T, dtype=torch.long)
        for b in bounds:
            seg[b:] += 1
        x = scene_vecs[seg] + 0.05 * torch.randn(T, D, generator=g)
        events = []
        for _ in range(q_per_video):
            n_ev = int(torch.randint(1, 4, (1,), generator=g))      # 1..3 key moments
            ev_vec = F.normalize(torch.randn(D, generator=g), dim=-1)
            pos = torch.randperm(T, generator=g)[:n_ev]
            x[pos] = x[pos] + 2.5 * ev_vec
            events.append((ev_vec, n_ev))
        feats.append(F.normalize(x, dim=-1))
        for j, (ev_vec, n_ev) in enumerate(events):
            qv = F.normalize(0.5 * ev_vec + torch.randn(D, generator=g) * 0.6, dim=-1)
            a = int(torch.randint(0, 5, (1,), generator=g))
            ov = F.normalize(torch.randn(5, D, generator=g), dim=-1)
            ov[a] = F.normalize(ev_vec + 0.8 * torch.randn(D, generator=g) / D ** 0.5, dim=-1)
            vid_idx.append(v); q.append(qv); opt.append(ov)
            qa.append(F.normalize(ov + 0.3 * qv, dim=-1)); ans.append(a)
            qtype.append(types[(n_ev - 1) * 3 % len(types) if n_ev > 1 else 5 + j % 3])
            qid.append(f"{v}_{j}")
    return {
        "video_ids": [str(i) for i in range(n_videos)],
        "video_feats": torch.stack(feats).half(),
        "vid_idx": torch.tensor(vid_idx), "q": torch.stack(q).half(),
        "opt": torch.stack(opt).half(), "qa": torch.stack(qa).half(),
        "answer": torch.tensor(ans), "qtype": qtype, "qid": qid,
        "meta": {"dummy": True},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="features_dummy")
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--train_videos", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for split, nv in [("train", args.train_videos), ("val", 300)]:
        d = make_split(nv, 3, args.n_frames, args.dim, g)
        torch.save(d, os.path.join(args.out_dir, f"{split}.pt"))
        print(split, len(d["answer"]), "questions")
