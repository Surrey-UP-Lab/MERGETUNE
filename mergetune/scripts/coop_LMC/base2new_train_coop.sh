#!/bin/bash

cd ../..

# custom config
DATA=MERGETUNE/mergetune/DATA
TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
SEED=$6
CFG=$7
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SEED in ${SEED}
do
    DIR=MERGETUNE/mergetune/output_KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    RESUME_COOP=MERGETUNE/mergetune/output_coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}
    
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
        TRAINER.COOP.W ${Clip_WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC ${COOP_LMC} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
