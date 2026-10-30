"""
Downstream video-QA model (stand-in for the paper's heavy VLM).

The paper supervises FFS with a large frozen VLM (EVA-CLIP-8B based). That does
not fit a student GPU budget, so we use a small transformer answerer over frozen
CLIP features. It is trained ONCE on random frame subsets (so it copes with any
number of input frames), then FROZEN while the selectors are trained, exactly
like the paper's setup. Every selection method is compared with this same
frozen model, so the comparison is fair.

Input : frames (B,T,D), mask (B,T) in {0,1} (or straight-through), qa (B,M,D)
        qa[:, j] = CLIP text embedding of "question + option j"
Output: logits (B,M)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamQA(nn.Module):
    def __init__(self, d_in=512, d_model=256, n_layers=2, n_heads=4, n_frames=32, dropout=0.1):
        super().__init__()
        self.v_proj = nn.Linear(d_in, d_model)
        self.t_proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, d_model))
        self.tok_type = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tok_type, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.sim_scale = nn.Parameter(torch.tensor(10.0))
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.05)))

    def forward(self, frames, mask, qa):
        B, T, _ = frames.shape
        M = qa.shape[1]
        m = mask.float()
        # scale frame tokens by the mask so gradients reach the selector (STE)
        v = (self.v_proj(frames.float()) + self.pos[:, :T] + self.tok_type[0]) * m.unsqueeze(-1)
        t = self.t_proj(qa.float()) + self.tok_type[1]                       # (B,M,d)
        v = v.unsqueeze(1).expand(B, M, T, -1).reshape(B * M, T, -1)
        t = t.reshape(B * M, 1, -1)
        x = torch.cat([t, v], 1)                                          # (B*M, 1+T, d)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=frames.device),
                         m.detach() < 0.5], 1)                            # unselected frames hidden
        pad = pad.unsqueeze(1).expand(B, M, T + 1).reshape(B * M, T + 1)
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        fused = self.score(x[:, 0]).view(B, M)

        # contrastive branch (like the paper's contrastive VLM): similarity between each
        # "question+option" text embedding and the selected frames, attention-pooled over
        # selected frames. Weights are m_t * exp(cos/temp), so it is differentiable in m.
        cos = torch.einsum("bmd,btd->bmt", F.normalize(qa.float(), dim=-1),
                           F.normalize(frames.float(), dim=-1))                  # (B,M,T)
        temp = self.log_temp.exp()
        w = m.unsqueeze(1) * torch.exp((cos - cos.detach().amax(-1, keepdim=True)) / temp)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        sim = (w * cos).sum(-1)                                                  # (B,M)
        return fused + self.sim_scale * sim


def uniform_mask(B, T, k, device):
    """Evenly spaced k of T frames (the standard default for VLMs)."""
    idx = torch.linspace(0, T - 1, steps=k * 2 + 1, device=device)[1::2].round().long() if k < T \
        else torch.arange(T, device=device)
    m = torch.zeros(B, T, device=device)
    m[:, idx] = 1.0
    return m


def random_subset_mask(B, T, k_min, k_max, device, p_all=0.1):
    """Random subsets of size U[k_min,k_max] (sometimes all frames) to train the downstream model."""
    ks = torch.randint(k_min, k_max + 1, (B,), device=device)
    allm = torch.rand(B, device=device) < p_all
    scores = torch.rand(B, T, device=device)
    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    m = (ranks < ks.unsqueeze(1)).float()
    m[allm] = 1.0
    return m
