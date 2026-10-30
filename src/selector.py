"""
Frame selectors.

FrameSelector implements, in one module:
  * Fixed top-k selection          (n_ctrl = 0, beta = 0)          -> ablation of FFS without [CTRL]
  * FFS (Buch et al., CVPR 2025)   (n_ctrl > 0, beta = 0)          -> baseline (our re-implementation)
  * FFS + Redundancy-Aware Selection (n_ctrl > 0, beta > 0)        -> OUR NOVELTY

The selection is a sequential, differentiable top-k:
  - forward pass  : hard greedy picks on Gumbel-perturbed logits (k picks)
  - backward pass : gradients flow through a relaxed k-hot vector (straight-through estimator)

With beta = 0 the greedy loop is exactly ordinary top-k, so our method strictly
generalises FFS. With beta > 0, after every pick the score of every remaining
frame is reduced by beta * (its similarity to the picked frame). [CTRL] tokens
are never penalised, so once the remaining frames are mostly repeats of what was
already picked, the [CTRL] tokens win the remaining slots and the model stops
early (fewer frames on redundant videos).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_like(x):
    u = torch.rand_like(x).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def frame_similarity(frames, n_ctrl, mode="centered"):
    """
    frames : (B, T, D) raw (frozen) CLIP frame features
    returns S : (B, T + n_ctrl, T + n_ctrl), non-negative, zero diagonal,
                zero rows/cols for [CTRL] tokens.
    mode   : 'raw'      -> S_ij = max(cos_ij, 0)
             'centered' -> S_ij = max(cos_ij - mean_offdiag(video), 0) / (1 - mean_offdiag)
                           (only frames that are MORE similar than usual count as repeats)
    """
    B, T, _ = frames.shape
    f = F.normalize(frames.float(), dim=-1)
    cos = f @ f.transpose(1, 2)                                   # (B,T,T)
    eye = torch.eye(T, device=frames.device, dtype=torch.bool).unsqueeze(0)
    if mode == "centered":
        off = cos.masked_fill(eye, 0.0).sum((1, 2), keepdim=True) / (T * (T - 1))
        S = ((cos - off) / (1.0 - off).clamp_min(1e-4)).clamp_min(0.0)
    elif mode == "raw":
        S = cos.clamp_min(0.0)
    else:
        raise ValueError(mode)
    S = S.masked_fill(eye, 0.0)
    if n_ctrl > 0:
        S = F.pad(S, (0, n_ctrl, 0, n_ctrl))                      # ctrl rows/cols = 0
    return S


def sequential_topk(logits, k, S=None, beta=0.0, tau=1.0, training=True, return_order=False, noise=None):
    """
    Differentiable (straight-through) greedy top-k with optional redundancy penalty.

    logits : (B, N)  selection scores for N = T + n_ctrl tokens
    S      : (B, N, N) similarity (or None)
    beta   : scalar tensor/float >= 0, redundancy strength
    returns ste_mask (B,N) [hard values forward, soft gradient backward], hard (B,N), soft (B,N)
    """
    B, N = logits.shape
    if noise is not None:
        g = noise
    else:
        g = gumbel_like(logits) if training else torch.zeros_like(logits)
    h = logits + g                       # hard path scores
    w = h / tau                          # soft path scores
    hard = torch.zeros_like(logits)
    soft = torch.zeros_like(logits)
    order = []
    use_pen = S is not None and (torch.is_tensor(beta) or beta > 0)
    neg_inf = torch.finfo(logits.dtype).min / 4
    for _ in range(k):
        # ---- soft (relaxed subset sampling, Xie & Ermon 2019) ----
        p = torch.softmax(w, dim=-1)
        soft = soft + p
        w = w + torch.log((1.0 - p).clamp_min(1e-6))
        if use_pen:
            w = w - (beta / tau) * torch.bmm(S, p.unsqueeze(-1)).squeeze(-1)
        # ---- hard (greedy argmax, never re-pick) ----
        idx = h.masked_fill(hard > 0, neg_inf).argmax(dim=-1)                    # (B,)
        order.append(idx)
        onehot = F.one_hot(idx, N).to(logits.dtype)
        hard = hard + onehot
        if use_pen:
            h = h - beta * torch.bmm(S, onehot.unsqueeze(-1)).squeeze(-1)
    soft = soft.clamp(0.0, 1.0)
    ste = hard - soft.detach() + soft
    if return_order:
        return ste, hard, soft, torch.stack(order, 1)        # order: (B,k) pick sequence
    return ste, hard, soft


class FrameSelector(nn.Module):
    """
    Lightweight multimodal selector (the 'FFS' policy).

    Inputs (pre-computed, frozen CLIP features):
        frames (B,T,D), question (B,D), options (B,M,D)
    Output dict:
        mask  (B,T) straight-through selection mask over real frames
        hard  (B,T) 0/1 selection
        n_sel (B,)  number of selected frames, in [k - n_ctrl, k]
    """

    def __init__(self, d_in=512, d_model=256, n_layers=2, n_heads=4, n_frames=32,
                 k=4, n_ctrl=3, beta=0.0, learn_beta=False, sim_mode="centered",
                 tau=1.0, dropout=0.1, decouple=True):
        super().__init__()
        self.decouple = decouple
        assert 0 <= n_ctrl < k, "need 0 <= n_ctrl < k  (k_low = k - n_ctrl >= 1)"
        self.k, self.n_ctrl, self.n_frames = k, n_ctrl, n_frames
        self.tau, self.sim_mode = tau, sim_mode

        self.v_proj = nn.Linear(d_in, d_model)
        self.t_proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, d_model))
        self.mod = nn.Parameter(torch.zeros(4, d_model))       # 0 video, 1 ctrl, 2 question, 3 option
        if n_ctrl > 0:
            self.ctrl = nn.Parameter(torch.zeros(1, n_ctrl, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mod, std=0.02)
        if n_ctrl > 0:
            nn.init.trunc_normal_(self.ctrl, std=0.02)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        # redundancy strength (our novelty). beta = softplus(raw) so it stays >= 0.
        self.use_redundancy = beta > 0 or learn_beta
        init = max(beta, 1e-3)
        raw = math.log(math.expm1(init))
        if learn_beta:
            self.beta_raw = nn.Parameter(torch.tensor(raw))
        else:
            self.register_buffer("beta_raw", torch.tensor(raw))

    @property
    def beta(self):
        return F.softplus(self.beta_raw) if self.use_redundancy else torch.zeros((), device=self.beta_raw.device)

    def forward(self, frames, question, options):
        B, T, _ = frames.shape
        v = self.v_proj(frames.float()) + self.pos[:, :T] + self.mod[0]
        toks = [v]
        if self.n_ctrl > 0:
            # [CTRL] init: mean visual embedding + learnable embedding (as in the paper)
            c = v.mean(1, keepdim=True) + self.ctrl + self.mod[1]
            toks.append(c)
        q = self.t_proj(question.float()).unsqueeze(1) + self.mod[2]
        o = self.t_proj(options.float()) + self.mod[3]
        toks += [q, o]
        x = self.norm(self.encoder(torch.cat(toks, 1)))
        sel_tok = x[:, : T + self.n_ctrl]
        logits = self.head(sel_tok).squeeze(-1)                         # (B, T + c)

        S = frame_similarity(frames, self.n_ctrl, self.sim_mode) if self.use_redundancy else None
        g = gumbel_like(logits) if self.training else torch.zeros_like(logits)
        ste, hard, soft = sequential_topk(logits, self.k, S, self.beta, self.tau, self.training, noise=g)
        mask = ste[:, :T]
        if self.decouple and self.n_ctrl > 0:
            # Gradient decoupling: the task loss only sees a relaxation over FRAMES (shift-invariant
            # w.r.t. the [CTRL] threshold), so it learns WHICH frames; HOW MANY is learned by the
            # count loss on [CTRL]. Forward values are unchanged (still the hard selection).
            Sf = S[:, :T, :T] if S is not None else None
            _, _, soft_f = sequential_topk(logits[:, :T], self.k, Sf, self.beta, self.tau,
                                           self.training, noise=g[:, :T])
            mask = hard[:, :T] - soft_f.detach() + soft_f
        return {
            "mask": mask, "hard": hard[:, :T], "soft": soft[:, :T],
            "logits": logits, "n_sel": hard[:, :T].sum(-1),
            "S": S,
        }


def greedy_frame_scores(frame_logits, k, S=None, beta=0.0):
    """
    Noise-free greedy pass over FRAMES only. Returns (scores, order), both (B,k):
    scores[:, j] = effective (penalised) score of the j-th pick at the moment it was picked.
    Differentiable w.r.t. frame_logits and beta. With beta = 0 this is just the sorted top-k.
    Scores are non-increasing in j, because penalties only ever lower scores.
    """
    B, T = frame_logits.shape
    h = frame_logits
    taken = torch.zeros(B, T, dtype=torch.bool, device=h.device)
    neg_inf = torch.finfo(h.dtype).min / 4
    scores, order = [], []
    use_pen = S is not None and (torch.is_tensor(beta) or beta > 0)
    for _ in range(k):
        idx = h.detach().masked_fill(taken, neg_inf).argmax(-1)
        scores.append(h.gather(1, idx[:, None]).squeeze(1))
        order.append(idx)
        taken = taken.scatter(1, idx[:, None], True)
        if use_pen:
            h = h - beta * S.gather(1, idx[:, None, None].expand(B, 1, T)).squeeze(1)
    return torch.stack(scores, 1), torch.stack(order, 1)


def count_hinge_loss(logits, T, k, target, margin=1.0, frame_scores=None):
    """
    Make the [CTRL] tokens act as a threshold that lets exactly t = target frames through.
    With sorted frame scores f_(1) >= f_(2) >= ... and sorted [CTRL] scores c_(1) >= c_(2) >= ...,
    top-k of the union contains exactly t frames iff
        f_(t)       > c_(k-t+1)   (t frames get in;        needed when k-t+1 <= c)
        c_(k-t)     > f_(t+1)     (the (t+1)-th is kept out; needed when t < k)
    We put a margin hinge on both. Unlike a softmax-based count, this never saturates.
    frame_scores: optional (B,T) scores after the redundancy penalty, in the selector's pick order.
    """
    B, N = logits.shape
    c = N - T
    f = (frame_scores if frame_scores is not None else logits[:, :T]).sort(-1, descending=True).values
    cs = logits[:, T:].sort(-1, descending=True).values
    t = target.long().clamp(k - c, k)
    loss = torch.zeros(B, device=logits.device)
    ar = torch.arange(B, device=logits.device)
    has_in = (k - t + 1) <= c                                   # (k-t+1)-th ctrl exists
    if has_in.any():
        fi = f[ar, (t - 1)]
        ci = cs[ar, (k - t).clamp(max=c - 1)]                   # index k-t  -> c_(k-t+1)
        loss = loss + torch.where(has_in, F.relu(margin - (fi - ci)), torch.zeros_like(fi))
    has_out = t < k
    if has_out.any():
        fo = f[ar, t.clamp(max=f.shape[1] - 1)]                          # f_(t+1)
        co = cs[ar, (k - t - 1).clamp(min=0)]                   # c_(k-t)
        loss = loss + torch.where(has_out, F.relu(margin - (co - fo)), torch.zeros_like(fo))
    return loss.mean()


@torch.no_grad()
def oracle_count(selector, out, frames, qa, answer, downstream):
    """
    Auxiliary target from a family of fixed-k policies (our reading of the paper's
    'supervise the selection behaviour with a set of fixed-k policies'):
    rank frames with the selector's OWN greedy order (no noise, no [CTRL]),
    then k* = smallest j in [k_low, k_high] such that the frozen downstream model
    answers correctly from the top-j frames; k* = k_high if none is correct.
    FFS and FFS+RAS are each supervised with their own ranking (RAS uses its penalty).
    """
    T = frames.shape[1]
    k_hi, k_lo = selector.k, selector.k - selector.n_ctrl
    S = out["S"][:, :T, :T] if out["S"] is not None else None
    _, order = greedy_frame_scores(out["logits"][:, :T].detach(), k_hi, S, selector.beta.detach())
    B = frames.shape[0]
    target = torch.full((B,), float(k_hi), device=frames.device)
    found = torch.zeros(B, dtype=torch.bool, device=frames.device)
    mask = torch.zeros(B, T, device=frames.device)
    for j in range(k_hi):
        mask.scatter_(1, order[:, j:j + 1], 1.0)
        if j + 1 < k_lo:
            continue
        ok = downstream(frames, mask, qa).argmax(-1) == answer
        target[ok & ~found] = float(j + 1)
        found |= ok
    return target


def redundancy_loss(mask, frames):
    """Optional auxiliary loss: average pairwise similarity among selected frames."""
    S = frame_similarity(frames, 0, "raw")
    m = mask
    pair = torch.einsum("bi,bij,bj->b", m, S, m)
    n = m.sum(-1)
    return (pair / (n * (n - 1)).clamp_min(1.0)).mean()
