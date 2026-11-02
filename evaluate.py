"""
Evaluate every method on the NExT-QA validation set with the SAME frozen downstream model.

    python evaluate.py --data_dir features --downstream checkpoints/downstream/downstream.pt \
                       --runs checkpoints --out_dir results

Methods:
  uniform@k   (k = 1,2,4,8,16,32)   evenly spaced frames (standard VLM default)
  random@4                          random 4 frames (sanity baseline)
  every trained selector found under --runs/<method>/seed*/selector.pt
     (fixed_topk, ffs, ffs_ras, ... ; several seeds are averaged -> mean ± std)

Outputs (in --out_dir):
  results.json, results_table.md, results_table.tex,
  acc_vs_frames.png, frame_hist.png, selections_<method>.json (for qualitative figures)
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from src.data import FeatureSplit
from src.downstream import uniform_mask
from src.selector import frame_similarity
from train_selector import build_selector, load_downstream
from src.utils import count_params, get_device, save_json, set_seed

GROUPS = ["Causal", "Temporal", "Descriptive"]


def selection_stats(frames, hard):
    """redundancy = mean pairwise cosine among selected frames; span = temporal coverage in [0,1]."""
    S = frame_similarity(frames, 0, "raw")
    n = hard.sum(-1)
    pair = torch.einsum("bi,bij,bj->b", hard, S, hard) / (n * (n - 1)).clamp_min(1.0)
    T = hard.shape[1]
    ar = torch.arange(T, device=hard.device).float()
    big = torch.where(hard > 0, ar, torch.full_like(ar, -1.0)).max(-1).values
    small = torch.where(hard > 0, ar, torch.full_like(ar, float(T))).min(-1).values
    span = (big - small) / max(T - 1, 1)
    return pair, n, span


@torch.no_grad()
def run_method(split, downstream, mask_fn, bs=256):
    preds, nsel, red, span, multi = [], [], [], [], []
    sels = []
    for b in split.iterate(bs):
        hard = mask_fn(b)
        logits = downstream(b["frames"], hard, b["qa"])
        preds.append(logits.argmax(-1).cpu())
        r, n, sp = selection_stats(b["frames"], hard)
        nsel.append(n.cpu()); red.append(r.cpu()); span.append(sp.cpu()); multi.append((n >= 2).cpu())
        sels += [torch.nonzero(h).flatten().tolist() for h in hard.cpu()]
    preds = torch.cat(preds); nsel = torch.cat(nsel); red = torch.cat(red); span = torch.cat(span)
    multi = torch.cat(multi)
    correct = (preds == split.answer.cpu()).float()
    res = {"acc": 100 * correct.mean().item(), "avg_frames": nsel.mean().item(),
           "redundancy": red[multi].mean().item() if multi.any() else float("nan"),
           "span": span.mean().item()}
    for gname in GROUPS:
        idx = torch.tensor([g == gname for g in split.group])
        if idx.any():
            res[f"acc_{gname}"] = 100 * correct[idx].mean().item()
            res[f"frames_{gname}"] = nsel[idx].mean().item()
    res["frame_hist"] = np.bincount(nsel.long().numpy(), minlength=split.n_frames + 1).tolist()
    per_q = [{"qid": q, "video_row": int(v), "pred": int(p), "answer": int(a), "frames": s}
             for q, v, p, a, s in zip(split.qid, split.vid_idx.cpu().tolist(), preds.tolist(),
                                     split.answer.cpu().tolist(), sels)]
    return res, per_q


def agg(list_of_res):
    keys = [k for k in list_of_res[0] if k != "frame_hist"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in list_of_res], dtype=float)
        out[k] = float(np.nanmean(v))
        out[k + "_std"] = float(np.nanstd(v)) if len(v) > 1 else 0.0
    out["frame_hist"] = np.mean([r["frame_hist"] for r in list_of_res], 0).tolist()
    out["n_seeds"] = len(list_of_res)
    return out


def fmt(r, k, nd=2):
    s = f"{r[k]:.{nd}f}"
    if r.get("n_seeds", 1) > 1:
        s += f" ± {r[k + '_std']:.{nd}f}"
    return s


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="features")
    ap.add_argument("--downstream", default="checkpoints/downstream/downstream.pt")
    ap.add_argument("--runs", default="checkpoints")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max_questions", type=int, default=None)
    args = ap.parse_args()
    set_seed(0)
    dev = get_device()
    split = FeatureSplit(os.path.join(args.data_dir, f"{args.split}.pt"), dev, args.max_questions)
    downstream = load_downstream(args.downstream, dev)
    T = split.n_frames
    print(f"{args.split}: {len(split)} questions, {T} frames/video")

    results, order = {}, []
    # ---- non-learned baselines ----
    for k in [1, 2, 4, 8, 16, 32]:
        if k > T:
            continue
        name = f"uniform@{k}"
        r, _ = run_method(split, downstream, lambda b, k=k: uniform_mask(len(b["answer"]), T, k, dev))
        results[name] = {**r, "n_seeds": 1, "kind": "uniform", "range": f"{k}"}
        order.append(name)
        print(f"{name:28s} acc {r['acc']:.2f}  frames {r['avg_frames']:.2f}")

    def rand4(b):
        sc = torch.rand(len(b["answer"]), T, device=dev)
        return (sc.argsort(1).argsort(1) < 4).float()
    r, _ = run_method(split, downstream, rand4)
    results["random@4"] = {**r, "n_seeds": 1, "kind": "random", "range": "4"}
    order.append("random@4")

    # ---- trained selectors ----
    sel_flops = {}
    pref = ["fixed_topk_nocue", "fixed_topk", "fixed_topk_ras", "ffs_nocue", "ffs", "ffs_ras", "ffs_ras_fixedbeta", "ffs_k8", "ffs_ras_k8"]
    mdirs = sorted(glob.glob(os.path.join(args.runs, "*")),
                   key=lambda p: (pref.index(os.path.basename(p)) if os.path.basename(p) in pref else 99, p))
    for mdir in mdirs:
        cks = sorted(glob.glob(os.path.join(mdir, "seed*", "selector.pt")))
        if not cks:
            continue
        name = os.path.basename(mdir)
        per_seed, perq_first = [], None
        for ck_path in cks:
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            sel = build_selector(ck["cfg"], ck["dim"], ck["n_frames"]).to(dev)
            sel.load_state_dict(ck["model"])
            sel.eval()

            def fn(b, sel=sel):
                return sel(b["frames"], b["q"], b["opt"], b["qa"])["hard"]
            r, perq = run_method(split, downstream, fn)
            r["beta"] = float(sel.beta.detach())
            per_seed.append(r)
            perq_first = perq_first or perq
        a = agg(per_seed)
        s = ck["cfg"]["selector"]
        a.update({"kind": "selector", "range": f"[{s['k'] - s['n_ctrl']},{s['k']}]",
                  "selector_params_M": count_params(sel) / 1e6})
        try:   # measure selector FLOPs for one sample
            from torch.utils.flop_counter import FlopCounterMode
            b = split.batch([0])
            with FlopCounterMode(display=False) as fc:
                sel(b["frames"], b["q"], b["opt"], b["qa"])
            a["selector_GFLOPs"] = fc.get_total_flops() / 1e9
        except Exception:
            pass
        results[name] = a
        order.append(name)
        save_json(perq_first, os.path.join(args.out_dir, f"selections_{name}.json"))
        print(f"{name:28s} acc {fmt(a,'acc')}  frames {fmt(a,'avg_frames')}  seeds {a['n_seeds']}")

    results["_meta"] = {"split": args.split, "n_questions": len(split), "frames_per_video": T,
                        "downstream_params_M": count_params(downstream) / 1e6}
    save_json(results, os.path.join(args.out_dir, "results.json"))

    # ---- markdown + latex tables ----
    hdr = ["Method", "Frames (range)", "Avg frames", "Acc", "Causal", "Temporal", "Descr.", "Redund.↓", "Rel. cost"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Method & Avg.\ frames & Acc & C & T & D & Redund.$\downarrow$ \\", r"\midrule"]
    for name in order:
        r = results[name]
        row = [name, r["range"], fmt(r, "avg_frames"), fmt(r, "acc"),
               *(f"{r.get('acc_'+g, float('nan')):.2f}" for g in GROUPS),
               f"{r['redundancy']:.3f}", f"{r['avg_frames']/T:.3f}"]
        md.append("| " + " | ".join(row) + " |")
        tex.append(f"{name.replace('_', '-')} & {r['avg_frames']:.2f} & {r['acc']:.2f} & "
                   + " & ".join(f"{r.get('acc_'+g, float('nan')):.1f}" for g in GROUPS)
                   + f" & {r['redundancy']:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    # compact table for the one-page report
    nice = {"uniform@4": "Uniform@4", f"uniform@{T}": f"Uniform@{T}",
            "fixed_topk_nocue": "Top-4 (FFS-style input)", "fixed_topk": r"\textbf{Top-4 + cues (ours)}",
            "fixed_topk_ras": r"\quad + RAS (ablation)",
            "ffs_nocue": "FFS (re-impl.)", "ffs": r"\textbf{FFS + cues (ours)}", "ffs_ras": r"\quad + RAS (ablation)"}
    ctex = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
            r"Method & Frames & Acc & C & T & D & Red.$\downarrow$ & Cost \\", r"\midrule"]
    for name in nice:
        if name not in results:
            continue
        r = results[name]
        acc = f"{r['acc']:.1f}" + (f"$\\pm${r['acc_std']:.1f}" if r.get("n_seeds", 1) > 1 else "")
        red = "--" if np.isnan(r["redundancy"]) else f"{r['redundancy']:.2f}"
        ctex.append(f"{nice[name]} & {r['avg_frames']:.2f} & {acc} & "
                    + " & ".join(f"{r.get('acc_'+g, float('nan')):.1f}" for g in GROUPS)
                    + f" & {red} & {r['avg_frames']/T:.3f} \\\\")
    ctex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out_dir, "results_table_compact.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(ctex) + "\n")
    with open(os.path.join(args.out_dir, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n\nRel. cost = avg downstream frames / " + str(T)
                + " (downstream compute scales ~linearly with frames).\n")
    with open(os.path.join(args.out_dir, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        uni = [n for n in order if n.startswith("uniform@")]
        ax.plot([results[n]["avg_frames"] for n in uni], [results[n]["acc"] for n in uni],
                "o-", color="#4a6fa5", label="uniform")
        for n in order:
            r = results[n]
            if r["kind"] == "selector":
                ax.errorbar(r["avg_frames"], r["acc"], yerr=r.get("acc_std", 0), fmt="s", ms=7, capsize=3,
                            label=n)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("avg. frames given to downstream model (log)")
        ax.set_ylabel(f"{args.split} accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "acc_vs_frames.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sels = [n for n in order if results[n]["kind"] == "selector"]
        w = 0.8 / max(len(sels), 1)
        for i, n in enumerate(sels):
            h = np.array(results[n]["frame_hist"]); h = h / h.sum()
            xs = np.arange(len(h))
            ax.bar(xs + i * w, h, width=w, label=n)
        mx = max([int(results[n]["range"].strip("[]").split(",")[1]) for n in sels] + [4])
        ax.set_xlim(0.5, mx + 1)
        ax.set_xlabel("# frames selected")
        ax.set_ylabel("fraction of questions")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "frame_hist.png"), dpi=200)
    except Exception as e:
        print("plotting skipped:", e)


if __name__ == "__main__":
    main()
