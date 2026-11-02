"""
Inference-time Redundancy-Aware Selection (RAS) on top of trained selectors (no retraining).

Scale-invariant RAS: frame scores are standardised within each video, z_i = (l_i - mean(l)) / std(l),
so the penalty beta * S_ij is measured in units of the selector's own score spread. (The learned-beta
version inside training left the picks unchanged because learned score gaps were ~10x larger than S.)

beta and k are chosen on held-out TRAINING videos (dev); the chosen setting is reported on val.

    python ras_sweep.py --features features --work work
"""
import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.data import FeatureSplit  # noqa: E402
from src.downstream import uniform_mask  # noqa: E402
from src.selector import frame_similarity, greedy_frame_scores  # noqa: E402
from train_selector import build_selector, load_downstream  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


@torch.no_grad()
def picks(sel, b, k, beta):
    T = b["frames"].shape[1]
    out = sel(b["frames"], b["q"], b["opt"], b["qa"])
    l = out["logits"][:, :T]
    z = (l - l.mean(1, keepdim=True)) / (l.std(1, keepdim=True) + 1e-6)
    S = frame_similarity(b["frames"], 0, "centered") if beta > 0 else None
    _, order = greedy_frame_scores(z, k, S, beta)
    return torch.zeros_like(l).scatter_(1, order, 1.0)


@torch.no_grad()
def evaluate(split, ds, mask_fn, bs=256):
    corr, red, n = 0, 0.0, 0
    for b in split.iterate(bs):
        m = mask_fn(b)
        corr += (ds(b["frames"], m, b["qa"]).argmax(-1) == b["answer"]).sum().item()
        S = frame_similarity(b["frames"], 0, "raw")
        cnt = m.sum(-1)
        r = torch.einsum("bi,bij,bj->b", m, S, m) / (cnt * (cnt - 1)).clamp_min(1)
        red += r[cnt >= 2].sum().item()
        n += (cnt >= 2).sum().item()
    return 100.0 * corr / len(split), (red / n if n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--work", default="work")
    ap.add_argument("--tag", default="EVA02-L-14")
    ap.add_argument("--method", default="fixed_topk", help="trained selector to re-use (its learned scores)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--max_questions", type=int, default=None)
    args = ap.parse_args()
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(args.work, f"checkpoints_{args.tag}")
    ds = load_downstream(os.path.join(ck, "downstream", "downstream.pt"), dev_)
    full = FeatureSplit(os.path.join(args.features, "train.pt"), dev_, args.max_questions)
    _, dv = full.split_by_video(0.1, 0)                       # same held-out training videos as training
    va = FeatureSplit(os.path.join(args.features, "val.pt"), dev_, args.max_questions)
    T = va.n_frames

    sels = []
    for p in sorted(glob.glob(os.path.join(ck, args.method, "seed*", "selector.pt"))):
        c = torch.load(p, map_location=dev_, weights_only=False)
        s = build_selector(c["cfg"], c["dim"], c["n_frames"]).to(dev_)
        s.load_state_dict(c["model"])
        s.eval()
        sels.append(s)
    print(f"{len(sels)} '{args.method}' selectors | dev {len(dv)} Qs (held-out train videos) | val {len(va)} Qs")

    rows = []
    for k in args.ks:
        ua, ur = evaluate(va, ds, lambda b, k=k: uniform_mask(len(b["answer"]), T, k, dev_))
        ud, _ = evaluate(dv, ds, lambda b, k=k: uniform_mask(len(b["answer"]), T, k, dev_))
        rows.append({"k": k, "beta": "uniform", "dev": ud, "val": ua, "val_std": 0.0, "red": ur})
        for beta in args.betas:
            dvs, vas, rds = [], [], []
            for s in sels:
                d, _ = evaluate(dv, ds, lambda b, s=s: picks(s, b, k, beta))
                v, r = evaluate(va, ds, lambda b, s=s: picks(s, b, k, beta))
                dvs.append(d); vas.append(v); rds.append(r)
            t = torch.tensor(vas)
            rows.append({"k": k, "beta": beta, "dev": sum(dvs) / len(dvs), "val": t.mean().item(),
                         "val_std": t.std().item() if len(vas) > 1 else 0.0, "red": sum(rds) / len(rds)})
            print(f"k={k} beta={beta:<5} dev {rows[-1]['dev']:.2f}  val {rows[-1]['val']:.2f} ± {rows[-1]['val_std']:.2f}"
                  f"  redund {rows[-1]['red']:.3f}", flush=True)

    print("\n| k | selection | dev acc (choose) | val acc | redundancy |\n|---|---|---|---|---|")
    for r in rows:
        name = "uniform" if r["beta"] == "uniform" else ("learned top-k" if r["beta"] == 0 else f"top-k + RAS β={r['beta']}")
        print(f"| {r['k']} | {name} | {r['dev']:.2f} | {r['val']:.2f} ± {r['val_std']:.2f} | {r['red']:.3f} |")
    print("\nChosen on dev (per k, RAS only, beta>0):")
    for k in args.ks:
        cand = [r for r in rows if r["k"] == k and r["beta"] not in ("uniform", 0.0)]
        base = [r for r in rows if r["k"] == k and r["beta"] == 0.0][0]
        best = max(cand, key=lambda r: r["dev"])
        print(f"  k={k}: beta={best['beta']}  val {best['val']:.2f} ± {best['val_std']:.2f}  vs learned top-k "
              f"{base['val']:.2f} ± {base['val_std']:.2f}   (redundancy {best['red']:.3f} vs {base['red']:.3f})")
    out = os.path.join(args.work, f"results_{args.tag}", f"ras_sweep_{args.method}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
