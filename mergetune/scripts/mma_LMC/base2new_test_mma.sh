#!/bin/bash

cd ../..

# custom config
DATA=DATA/
TRAINER=MMA_LMC
DATASET=$1
LOSS_TYPE=cosine
COOP_LMC=True
W_LMC=$2
CFG=$3
SEED=$4
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

SUB=new

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output_MMA_LMC/base2new/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    DIR=output_MMA_LMC/base2new/evaluate_valbest/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        PYTHONPATH=Dassl.ProGrad.pytorch:$PYTHONPATH \
        python MERGETUNE/mergetune/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
        --config-file MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done