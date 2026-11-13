#!/bin/bash
# AnomalyDINO runner (env lives in /tmp/yzhang176; recreate with setup_env.sh if /tmp was wiped)
# usage: bash run_ad.sh [clf|segm] "1 2 4 8" [num_seeds] [extra args...]
set -e
W=/tmp/yzhang176
MODE=${1:-clf}; SHOTS=${2:-1}; SEEDS=${3:-1}; shift 3 2>/dev/null || true
export TORCH_HOME=$W/cache/torch
export MPLCONFIGDIR=$W/cache/mpl
mkdir -p $MPLCONFIGDIR
cd $W/src/AnomalyDINO-main
ARGS="--dataset VisA --data_root $W/data/VisA_pytorch/1cls --shots $SHOTS --num_seeds $SEEDS --preprocess agnostic --no-save_examples"
if [ "$MODE" = "segm" ]; then
  echo "[warn] segm mode writes ~11G of full-res maps for the 6-class set; clean results_VisA after each run"
  ARGS="$ARGS --eval_segm"
fi
$W/envs/ad/bin/python run_anomalydino.py $ARGS "$@"
