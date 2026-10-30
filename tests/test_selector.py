"""Quick correctness checks:  python -m pytest -q tests   (or python tests/test_selector.py)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.selector import FrameSelector, frame_similarity, sequential_topk  # noqa: E402
from src.downstream import DownstreamQA  # noqa: E402


def test_beta_zero_equals_topk():
    torch.manual_seed(0)
    l = torch.randn(8, 35)
    _, hard, _ = sequential_topk(l, 4, None, 0.0, training=False)
    ref = torch.zeros_like(l).scatter_(1, l.topk(4, -1).indices, 1.0)
    assert torch.equal(hard, ref)


def test_selection_range_and_grad():
    torch.manual_seed(0)
    B, T, D = 6, 32, 64
    for learn in [False, True]:
        sel = FrameSelector(D, 32, 1, 2, T, k=4, n_ctrl=3, beta=1.0 if learn else 0.0, learn_beta=learn)
        f, q, o = torch.randn(B, T, D), torch.randn(B, D), torch.randn(B, 5, D)
        sel.train()
        out = sel(f, q, o)
        assert ((out["n_sel"] >= 1) & (out["n_sel"] <= 4)).all()
        ds = DownstreamQA(D, 32, 1, 2, T)
        loss = ds(f, out["mask"], torch.randn(B, 5, D)).logsumexp(-1).mean()
        loss.backward()
        assert sel.head[0].weight.grad is not None and sel.head[0].weight.grad.abs().sum() > 0
        if learn:
            assert sel.beta_raw.grad is not None


def test_redundancy_skips_duplicates():
    # frames 0..3 are near-identical and score highest; frames 10 and 20 are distinct but slightly lower.
    T, D = 32, 16
    f = torch.randn(1, T, D) * 0.01
    base = torch.randn(D)
    f[0, :4] += base
    f[0, 10] += torch.randn(D)
    f[0, 20] += torch.randn(D)
    for t in range(T):
        if t not in (0, 1, 2, 3, 10, 20):
            f[0, t] += torch.randn(D)
    l = torch.full((1, T), -5.0)
    l[0, :4] = torch.tensor([3.0, 2.9, 2.8, 2.7])
    l[0, 10], l[0, 20] = 2.0, 1.9
    S = frame_similarity(f, 0, "centered")
    _, h0, _ = sequential_topk(l, 3, S, 0.0, training=False)
    _, h1, _ = sequential_topk(l, 3, S, 5.0, training=False)
    assert torch.nonzero(h0[0]).flatten().tolist() == [0, 1, 2]          # plain top-k: duplicates
    assert torch.nonzero(h1[0]).flatten().tolist() == [0, 10, 20]        # ours: diverse


def test_ctrl_wins_when_redundant():
    # with [CTRL] slightly below the duplicates, plain FFS picks 3 duplicates, ours stops after 1
    T, D, c = 8, 16, 2
    f = torch.randn(1, T, D) * 0.01
    f[0, :3] += torch.randn(D)
    for t in range(3, T):
        f[0, t] += torch.randn(D)
    l = torch.full((1, T + c), -5.0)
    l[0, :3] = torch.tensor([3.0, 2.9, 2.8])
    l[0, T:] = 2.5
    S = frame_similarity(f, c, "centered")
    _, h0, _ = sequential_topk(l, 3, S, 0.0, training=False)
    _, h1, _ = sequential_topk(l, 3, S, 5.0, training=False)
    assert h0[0, :T].sum() == 3 and h1[0, :T].sum() == 1


def test_count_hinge_reaches_target():
    from src.selector import count_hinge_loss, greedy_frame_scores
    torch.manual_seed(0)
    T, k, c = 32, 4, 3
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    l = torch.randn(4, T + c).requires_grad_()
    opt = torch.optim.SGD([l], lr=0.5)
    for _ in range(300):
        fs, _ = greedy_frame_scores(l[:, :T], k)
        loss = count_hinge_loss(l, T, k, target, frame_scores=fs)
        opt.zero_grad(); loss.backward(); opt.step()
    _, hard, _ = sequential_topk(l.detach(), k, None, 0.0, training=False)
    assert hard[:, :T].sum(-1).tolist() == target.tolist(), hard[:, :T].sum(-1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
