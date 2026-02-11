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

for SEED in ${SEED}
do
    DIR=output_MMA_LMC/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    RESUME=output_MMA/base2new/train_base/${DATASET}/shots_16/MultiModalAdapter/seed${SEED}
    
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
        --resume-coop ${RESUME} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC ${COOP_LMC} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
