"""
Stage 2: train a frame selector against the FROZEN downstream model.

    python train_selector.py --config configs/ffs.yaml        # baseline FFS  (k=4, 3 [CTRL] -> 1..4 frames)
    python train_selector.py --config configs/ffs_ras.yaml    # OURS: FFS + Redundancy-Aware Selection
    python train_selector.py --config configs/fixed_topk.yaml # ablation: no [CTRL], always 4 frames

Loss = CrossEntropy(downstream answer)                       (task loss, backprop through STE mask)
     + lambda_count  * hinge([CTRL] threshold lets k* frames in) (aux: k* from fixed-k policies, see src/selector.py)
     + lambda_budget * mean(#selected / k)                   (optional, off by default)
     + lambda_red    * pairwise redundancy of selection       (optional, off by default)
"""
import argparse
import os

import torch
import torch.nn.functional as F

from src.data import FeatureSplit
from src.downstream import DownstreamQA
from src.selector import (FrameSelector, count_hinge_loss, greedy_frame_scores, oracle_count,
                          redundancy_loss)
from src.utils import Logger, count_params, get_device, load_yaml, save_json, set_seed


def load_downstream(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    c = ck["cfg"]
    m = DownstreamQA(ck["dim"], c["d_model"], c["n_layers"], c["n_heads"], ck["n_frames"], c["dropout"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_selector(cfg, dim, n_frames):
    s = cfg["selector"]
    return FrameSelector(dim, s["d_model"], s["n_layers"], s["n_heads"], n_frames,
                         k=s["k"], n_ctrl=s["n_ctrl"], beta=s.get("beta", 0.0),
                         learn_beta=s.get("learn_beta", False), sim_mode=s.get("sim_mode", "centered"),
                         tau=s.get("tau", 1.0), dropout=s.get("dropout", 0.1),
                         decouple=s.get("decouple", True))


@torch.no_grad()
def evaluate(selector, downstream, split, bs=256):
    selector.eval()
    correct, frames = 0, 0.0
    for b in split.iterate(bs):
        out = selector(b["frames"], b["q"], b["opt"])
        pred = downstream(b["frames"], out["hard"], b["qa"]).argmax(-1)
        correct += (pred == b["answer"]).sum().item()
        frames += out["n_sel"].sum().item()
    return 100.0 * correct / len(split), frames / len(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--downstream", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_questions", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for k in ["data_dir", "out_dir", "downstream", "epochs", "seed"]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["out_dir"] = os.path.join(cfg["out_dir"], f"seed{cfg['seed']}")

    set_seed(cfg["seed"])
    dev = get_device(cfg.get("device", "auto"))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log = Logger(os.path.join(cfg["out_dir"], "train_log.txt"))
    log("config:", cfg)

    full = FeatureSplit(os.path.join(cfg["data_dir"], "train.pt"), dev, args.max_questions)
    tr, dv = full.split_by_video(cfg["dev_frac"], cfg["split_seed"])
    log(f"train {len(tr)} | dev (held-out train videos) {len(dv)}")

    downstream = load_downstream(cfg["downstream"], dev)
    sel = build_selector(cfg, tr.dim, tr.n_frames).to(dev)
    log(f"selector params: {count_params(sel)/1e6:.2f} M | downstream params: {count_params(downstream)/1e6:.2f} M")
    log(f"selection range: [{sel.k - sel.n_ctrl}, {sel.k}] frames from {tr.n_frames}")

    opt = torch.optim.AdamW(sel.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * ((len(tr) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(cfg["seed"])
    lam_b, lam_r = cfg.get("lambda_budget", 0.0), cfg.get("lambda_red", 0.0)
    lam_c = cfg.get("lambda_count", 0.0)

    best, hist = -1.0, []
    for ep in range(cfg["epochs"]):
        sel.train()
        tot, nsel, n = 0.0, 0.0, 0
        for b in tr.iterate(cfg["batch_size"], shuffle=True, generator=g):
            out = sel(b["frames"], b["q"], b["opt"])
            logits = downstream(b["frames"], out["mask"], b["qa"])
            loss = F.cross_entropy(logits, b["answer"])
            if lam_c > 0 and sel.n_ctrl > 0:
                # aux: soft frame count -> k* from fixed-k policies (identical for FFS and ours)
                T = b["frames"].shape[1]
                tgt = oracle_count(sel, out, b["frames"], b["qa"], b["answer"], downstream)
                S = out["S"][:, :T, :T] if out["S"] is not None else None
                fs, _ = greedy_frame_scores(out["logits"][:, :T], sel.k, S, sel.beta)
                loss = loss + lam_c * count_hinge_loss(out["logits"], T, sel.k, tgt, frame_scores=fs)
            if lam_b > 0:
                # differentiable expected number of selected frames (soft k-hot), normalised by k
                loss = loss + lam_b * out["soft"].sum(-1).mean() / sel.k
            if lam_r > 0:
                loss = loss + lam_r * redundancy_loss(out["mask"], b["frames"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sel.parameters(), 1.0)
            opt.step()
            sched.step()
            B = len(b["answer"])
            tot += loss.item() * B
            nsel += out["n_sel"].sum().item()
            n += B
        acc, avgf = evaluate(sel, downstream, dv)
        beta = float(sel.beta)
        hist.append({"epoch": ep + 1, "loss": tot / n, "train_avg_frames": nsel / n,
                     "dev_acc": acc, "dev_avg_frames": avgf, "beta": beta})
        log(f"epoch {ep+1:02d} | loss {tot/n:.4f} | train frames {nsel/n:.2f} | "
            f"dev acc {acc:.2f} | dev frames {avgf:.2f} | beta {beta:.3f}")
        if acc > best:
            best = acc
            torch.save({"model": sel.state_dict(), "cfg": cfg, "n_frames": tr.n_frames, "dim": tr.dim},
                       os.path.join(cfg["out_dir"], "selector.pt"))
    save_json(hist, os.path.join(cfg["out_dir"], "history.json"))
    log(f"best dev acc {best:.2f} | saved {os.path.join(cfg['out_dir'], 'selector.pt')}")


if __name__ == "__main__":
    main()
