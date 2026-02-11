#!/bin/bash

cd ../..

# custom config
DATA=MERGETUNE/mergetune/DATA
TRAINER=CoOp

DATASET=$1
SEED=$2
CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=5  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SEED in ${SEED}
do
    DIR=MERGETUNE/mergetune/output_coop/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python MERGETUNE/mergetune/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
        --config-file MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
