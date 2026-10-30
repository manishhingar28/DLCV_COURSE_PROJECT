"""
Offline test of scripts/extract_features.py: tiny fake videos in a zip, NExT-QA-style CSVs,
and a stand-in for the CLIP model (so no internet is needed).   python tests/test_extract.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


class FakeCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 512)

    def encode_image(self, x):
        return self.lin(x.mean((2, 3)))

    def encode_text(self, t):
        return torch.randn(t.shape[0], 512)


def main():
    import cv2
    import open_clip
    tmp = tempfile.mkdtemp()
    ann = os.path.join(tmp, "ann"); os.makedirs(ann)
    vids = ["1001", "1002", "1003"]
    vmap = {v: f"0000/9{v}" for v in vids}            # videos are stored under VidOR ids
    json.dump(vmap, open(os.path.join(ann, "map_vid_vidorID.json"), "w"))
    zpath = os.path.join(tmp, "NExTVideo.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for v in vids[:2] + ["9999"]:                 # video 1003 is deliberately missing
            p = os.path.join(tmp, f"9{v}.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for i in range(40):
                w.write(np.full((48, 64, 3), i * 5, np.uint8))
            w.release()
            z.write(p, f"NExTVideo/0000/9{v}.mp4")
    hdr = ["video", "frame_count", "width", "height", "question", "answer", "qid", "type",
           "a0", "a1", "a2", "a3", "a4"]
    for split in ["train", "val"]:
        with open(os.path.join(ann, f"{split}.csv"), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(hdr)
            for i, v in enumerate(vids):
                for q in range(2):
                    wr.writerow([v, 40, 64, 48, f"why did x {q}", q % 5, q, "CW" if q else "DO",
                                 "a", "b", "c", "d", "e"])
    open_clip.create_model_and_transforms = lambda *a, **k: (
        FakeCLIP(), None,
        lambda im: torch.from_numpy(np.asarray(im.resize((32, 32)), np.float32)).permute(2, 0, 1) / 255)
    open_clip.get_tokenizer = lambda *a, **k: (lambda texts: torch.zeros(len(texts), 77, dtype=torch.long))
    import extract_features
    out = os.path.join(tmp, "feats")
    sys.argv = ["x", "--video_zip", zpath, "--ann_dir", ann, "--out_dir", out, "--n_frames", "8"]
    extract_features.main()
    d = torch.load(os.path.join(out, "val.pt"), weights_only=False)
    assert d["video_feats"].shape == (2, 8, 512), d["video_feats"].shape
    assert len(d["answer"]) == 4 and d["opt"].shape == (4, 5, 512) and d["qa"].shape == (4, 5, 512)
    assert d["video_ids"] == ["1001", "1002"]
    from src.data import FeatureSplit
    s = FeatureSplit(os.path.join(out, "train.pt"))
    b = s.batch([0, 3])
    assert b["frames"].shape == (2, 8, 512)
    # resume: second run must use the cache
    extract_features.main()
    shutil.rmtree(tmp)
    print("ok test_extract")


if __name__ == "__main__":
    main()
