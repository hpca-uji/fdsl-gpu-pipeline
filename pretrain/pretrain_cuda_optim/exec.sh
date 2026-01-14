#!/bin/bash

## Parameters ##

# Dataset
DATASET_CFG="../config/classes.cfg"
DATASET_SIZE=300000256
RES=224
KERNEL_RES=512
DATASET_WORKERS=8

# Model Parameters
MODEL="deit_tiny_patch16_224"
#MODEL="deit_base_patch16_224"
#MODEL="vit_large_patch16_224"
NUM_CLASSES=1000

# Training parameters
WARMUP_ITERS=5000
LR=0.001
OPT="adamw"
WEIGHT_DECAY=0.05
BATCH_SIZE=256
SCHED="step"

# Data augmentation parameters
NUM_OPS=2
MAGNITUDE=28
AUG_REPEATS=1
MIXUP=0.8
CUTMIX=1.0 
REPROB=0.25
DROP_PATH=0.1
SMOOTHING=0.1

# Pretrain
torchrun --nproc_per_node=1 pretrain.py \
    --dataset-cfg-path $DATASET_CFG --dataset-size $DATASET_SIZE -j $DATASET_WORKERS \
    --res $RES --kernel-res $KERNEL_RES --experiment optim \
    --model $MODEL --num-classes $NUM_CLASSES --amp --log-interval 50 \
    --warmup-iters $WARMUP_ITERS --lr $LR --opt $OPT --weight-decay $WEIGHT_DECAY -b $BATCH_SIZE --sched $SCHED \
    --num-ops $NUM_OPS --magnitude $MAGNITUDE --aug-repeats $AUG_REPEATS \
    --mixup $MIXUP --cutmix $CUTMIX --reprob $REPROB --drop-path $DROP_PATH --smoothing $SMOOTHING &> progress.txt