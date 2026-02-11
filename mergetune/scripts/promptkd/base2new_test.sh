#!/bin/bash

cd ../..

# Original PromptKD Base-to-Novel Evaluation Script
# This script evaluates the original PromptKD model on novel classes

# custom config
DATA=DATA/
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0
# LOADEP=20  # Load epoch (adjust based on your training epochs)
SUB=new    # Test on 'new' (novel) classes

# Directory structure
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
MODEL_DIR=output_PromptKD/base2new/train_base/${COMMON_DIR}
DIR=output_PromptKD/base2new/test_${SUB}/${COMMON_DIR}

echo "Original PromptKD Base-to-Novel Evaluation"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Model directory: ${MODEL_DIR}"
echo "Output directory: ${DIR}"
echo "Testing on: ${SUB} classes"

# Check if evaluation has already been done
if [ -d "$DIR" ]; then
    echo "Evaluating model (results directory exists)"
    echo "Results are available in ${DIR}. Resuming..."
else
    echo "Evaluating model (first time)"
    echo "Running evaluation and saving output to ${DIR}"
fi

# Run the evaluation
CUDA_VISIBLE_DEVICES=0 python MERGETUNE/mergetune/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
    --config-file MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0

echo "Evaluation completed!"
echo "Results saved in: ${DIR}"
