#!/bin/bash

cd ../..

# custom config
DATA=DATA/
TRAINER=KgCoOp_COOP_LMC
DATASET=$1
W=$2
W_LMC=$3
SEED=$4
LOSS_TYPE=cosine
COOP_LMC=True
CFG=$5
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
NUM_SAMPLES=5

for SEED in $SEED
do
    DIR=output_KgCoOp_LMC/KgCoOp/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    RESUME_COOP=output_KgCoop/base2new/train_base/${DATASET}/shots_16_8.0/KgCoOp/vit_b16_ep100_ctxv1/seed${SEED}
    
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
    --resume-coop ${RESUME_COOP} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.COOP.COOP_LMC ${COOP_LMC} \
    TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
    DATASET.SUBSAMPLE_CLASSES base
    fi
done
