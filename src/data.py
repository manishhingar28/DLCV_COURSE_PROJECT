"""
Feature-file format (one file per split, produced by scripts/extract_features.py):

    {
      "video_ids": [str] * V,
      "video_feats": FloatTensor/HalfTensor (V, T, D)   # L2-normalised CLIP image features, T frames per video
      "vid_idx":  LongTensor (N,)                       # row in video_feats for each question
      "q":        (N, D)                                # CLIP text feature of the question
      "opt":      (N, M, D)                             # CLIP text feature of each answer option
      "qa":       (N, M, D)                             # CLIP text feature of "question? option"
      "answer":   LongTensor (N,)
      "qtype":    [str] * N                             # NExT-QA type, e.g. CW, CH, TN, TC, TP, DO, DL, DC
      "qid":      [str] * N
    }
Everything fits in memory (NExT-QA train ~ 130 MB in fp16), so we batch by indexing.
"""
import torch

TYPE_GROUP = {"CW": "Causal", "CH": "Causal",
              "TN": "Temporal", "TC": "Temporal", "TP": "Temporal",
              "DO": "Descriptive", "DL": "Descriptive", "DC": "Descriptive"}


class FeatureSplit:
    def __init__(self, path, device="cpu", max_questions=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        n = len(d["answer"]) if max_questions is None else min(max_questions, len(d["answer"]))
        self.video_feats = d["video_feats"].float().to(device)
        self.vid_idx = d["vid_idx"][:n].to(device)
        self.q = d["q"][:n].float().to(device)
        self.opt = d["opt"][:n].float().to(device)
        self.qa = d["qa"][:n].float().to(device)
        self.answer = d["answer"][:n].to(device)
        self.qtype = list(d["qtype"][:n])
        self.qid = list(d["qid"][:n])
        self.group = [TYPE_GROUP.get(t, "Other") for t in self.qtype]
        self.device = device

    def __len__(self):
        return len(self.answer)

    @property
    def n_frames(self):
        return self.video_feats.shape[1]

    @property
    def dim(self):
        return self.video_feats.shape[2]

    def batch(self, idx):
        idx = torch.as_tensor(idx, device=self.device)
        return {
            "idx": idx,
            "frames": self.video_feats[self.vid_idx[idx]],
            "q": self.q[idx], "opt": self.opt[idx], "qa": self.qa[idx],
            "answer": self.answer[idx],
        }

    def subset(self, rows):
        """Shallow copy restricted to question rows (video features are shared)."""
        import copy
        s = copy.copy(self)
        rows_t = torch.as_tensor(rows, dtype=torch.long)
        r = rows_t.to(self.device)
        s.vid_idx, s.q, s.opt, s.qa, s.answer = (self.vid_idx[r], self.q[r], self.opt[r],
                                                 self.qa[r], self.answer[r])
        rl = rows_t.tolist()
        s.qtype = [self.qtype[i] for i in rl]
        s.qid = [self.qid[i] for i in rl]
        s.group = [self.group[i] for i in rl]
        return s

    def split_by_video(self, dev_frac=0.1, seed=0):
        """Hold out whole videos (no leakage) as a dev set for checkpoint selection."""
        vids = torch.unique(self.vid_idx).cpu()
        g = torch.Generator().manual_seed(seed)
        vids = vids[torch.randperm(len(vids), generator=g)]
        n_dev = max(1, int(round(dev_frac * len(vids))))
        dev_set = set(vids[:n_dev].tolist())
        is_dev = [v in dev_set for v in self.vid_idx.cpu().tolist()]
        tr_rows = [i for i, d in enumerate(is_dev) if not d]
        dv_rows = [i for i, d in enumerate(is_dev) if d]
        return self.subset(tr_rows), self.subset(dv_rows)

    def iterate(self, batch_size, shuffle=False, generator=None):
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for s in range(0, n, batch_size):
            yield self.batch(order[s:s + batch_size])
