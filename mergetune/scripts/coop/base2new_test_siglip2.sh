#!/bin/bash

cd ../..

# custom config
DATA=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/DATA
TRAINER=CoOp_SigLIP2
DATASET=$1
CFG=siglip2_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=5  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

LOADEP=100
SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/output_coop_siglip2/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    DIR=/mnt/fast/nobackup/scratch4weeks/ww00620/wwq/MERGETUNE/mergetune/output_coop_siglip2/evaluate/test_${SUB}/${COMMON_DIR}


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
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
