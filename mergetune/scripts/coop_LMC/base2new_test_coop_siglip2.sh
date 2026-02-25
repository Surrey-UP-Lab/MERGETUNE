#!/bin/bash

cd ../..

# custom config
DATA=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/DATA
TRAINER=KgCoOp_COOP_LMC_SigLIP2
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
CFG=$6
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/output_KgCoOp_COOP_LMC_SigLIP2/CoOp_SigLIP2/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    DIR=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/output_KgCoOp_COOP_LMC_SigLIP2/CoOp_SigLIP2/evaluate/test_${SUB}/${COMMON_DIR}
    RESUME_COOP=None


    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python /mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file /mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
        --config-file /mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only-each-10-epoch \
        --max-epoch 100 \
        --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

