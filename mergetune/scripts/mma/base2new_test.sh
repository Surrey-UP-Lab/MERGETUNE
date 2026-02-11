#!/bin/bash

cd ../..

DATA=DATA/
TRAINER=MultiModalAdapter

DATASET=$1
SEED=$2
CFG=$3
SHOTS=16
LOADEP=$4
SUB=new


COMMON_DIR=${DATASET}_new/shots_${SHOTS}/${TRAINER}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python MERGETUNE/mergetune/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
    --config-file MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi