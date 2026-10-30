#!/usr/bin/env bash
# Full pipeline after features exist.  Usage: bash scripts/run_all.sh <features_dir> <checkpoint_dir> <results_dir> [seeds...]
set -e
FEAT=${1:-features}; CK=${2:-checkpoints}; RES=${3:-results}; shift 3 || true
SEEDS=${@:-0 1 2}
python train_downstream.py --config configs/downstream.yaml --data_dir $FEAT --out_dir $CK/downstream
for s in $SEEDS; do
  for m in fixed_topk ffs ffs_ras; do
    python train_selector.py --config configs/$m.yaml --data_dir $FEAT --downstream $CK/downstream/downstream.pt \
        --out_dir $CK/$m --seed $s
  done
done
python evaluate.py --data_dir $FEAT --downstream $CK/downstream/downstream.pt --runs $CK --out_dir $RES
