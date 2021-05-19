#!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=4 train.py configs/swin/saliency_swin_tiny.py --launcher pytorch
