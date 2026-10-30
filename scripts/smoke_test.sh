#!/usr/bin/env bash
# CPU-only end-to-end check on synthetic features (~10 min). Not for reporting results.
set -e
python tests/test_selector.py
python tests/test_extract.py
python scripts/make_dummy_data.py --out_dir features_dummy
python train_downstream.py --config configs/smoke/downstream.yaml --epochs 3
for m in fixed_topk ffs ffs_ras; do python train_selector.py --config configs/smoke/$m.yaml --epochs 2; done
python evaluate.py --data_dir features_dummy --downstream runs_dummy/downstream/downstream.pt --runs runs_dummy --out_dir results_dummy
