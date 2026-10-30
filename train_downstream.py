"""
Stage 1: train the downstream QA model on random frame subsets, then freeze it.

    python train_downstream.py --config configs/downstream.yaml
"""
import argparse
import os

import torch
import torch.nn.functional as F

from src.data import FeatureSplit
from src.downstream import DownstreamQA, random_subset_mask, uniform_mask
from src.utils import Logger, count_params, get_device, load_yaml, save_json, set_seed


@torch.no_grad()
def evaluate_uniform(model, split, k, bs=256):
    model.eval()
    correct = 0
    for b in split.iterate(bs):
        m = uniform_mask(len(b["answer"]), split.n_frames, k, split.device)
        correct += (model(b["frames"], m, b["qa"]).argmax(-1) == b["answer"]).sum().item()
    return 100.0 * correct / len(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/downstream.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_questions", type=int, default=None, help="debug: use a subset")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for k in ["data_dir", "out_dir", "epochs", "seed"]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    set_seed(cfg["seed"])
    dev = get_device(cfg.get("device", "auto"))
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "train_log.txt"))
    log("config:", cfg)

    full = FeatureSplit(os.path.join(cfg["data_dir"], "train.pt"), dev, args.max_questions)
    tr, va = full.split_by_video(cfg["dev_frac"], cfg["split_seed"])   # 'va' = held-out dev, NOT the val set
    log(f"train {len(tr)} | dev (held-out train videos) {len(va)} | frames {tr.n_frames} | dim {tr.dim}")

    model = DownstreamQA(tr.dim, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], tr.n_frames,
                         cfg["dropout"]).to(dev)
    log(f"downstream params: {count_params(model)/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * ((len(tr) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(cfg["seed"])

    best, hist = -1.0, []
    for ep in range(cfg["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for b in tr.iterate(cfg["batch_size"], shuffle=True, generator=g):
            m = random_subset_mask(len(b["answer"]), tr.n_frames, cfg["k_min"], cfg["k_max"], dev,
                                   cfg["p_all_frames"])
            loss = F.cross_entropy(model(b["frames"], m, b["qa"]), b["answer"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b["answer"])
            n += len(b["answer"])
        acc4 = evaluate_uniform(model, va, 4)
        acc32 = evaluate_uniform(model, va, tr.n_frames)
        hist.append({"epoch": ep + 1, "loss": tot / n, "dev_uniform4": acc4, "dev_uniform_all": acc32})
        log(f"epoch {ep+1:02d} | loss {tot/n:.4f} | dev acc uniform@4 {acc4:.2f} | uniform@{tr.n_frames} {acc32:.2f}")
        score = acc4 + acc32
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": cfg, "n_frames": tr.n_frames, "dim": tr.dim},
                       os.path.join(out, "downstream.pt"))
    save_json(hist, os.path.join(out, "history.json"))
    log("saved", os.path.join(out, "downstream.pt"))


if __name__ == "__main__":
    main()
