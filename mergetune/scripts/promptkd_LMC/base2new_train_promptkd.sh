cd ../..

# custom config
DATA=DATA/
TRAINER=PromptKD_LMC

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2
PRETRAINED_STUDENT_PATH=$3 # Path to pretrained student model
KD_WEIGHT=$4

CFG=vit_b16_c2_ep20_batch8_4+4ctx_lmc
SHOTS=0


# LMC hyperparameters
LAMBDA_REG=${4:-100.0}   # L2 regularization weight (default: 100.0, increased for meaningful impact)
BETA_LMC=${5:-50.0}      # LMC connectivity weight (default: 50.0, increased for meaningful impact)
NUM_SAMPLES=${6:-10}     # Number of interpolation samples (default: 10)

DIR=output_PromptKD_LMC/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${LAMBDA_REG}_${BETA_LMC}_${NUM_SAMPLES}_${KD_WEIGHT}/${CFG}/seed_${SEED}


# Check if pretrained student model path is provided and exists
if [ -z "$PRETRAINED_STUDENT_PATH" ]; then
    echo "Error: Please provide the path to pretrained student model as the 3rd argument"
    echo "Usage: $0 <dataset> <seed> <pretrained_student_path> [lambda_reg] [beta_lmc] [num_samples]"
    echo "Example: $0 caltech101 1 output/base2new/train_base/caltech101/shots_0/PromptKD/vit_b16_c2_ep20_batch8_4+4ctx/seed_1"
    exit 1
fi

if [ ! -d "$PRETRAINED_STUDENT_PATH" ]; then
    echo "Error: Pretrained student model path does not exist: $PRETRAINED_STUDENT_PATH"
    echo "Please check the path and try again."
    exit 1
fi

# Check if the actual model file exists
MODEL_FILE="${PRETRAINED_STUDENT_PATH}/VLPromptLearner/model-best.pth.tar"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Pretrained student model file not found: $MODEL_FILE"
    echo "Please ensure you have a trained PromptKD model at this location."
    exit 1
fi

echo "Training with LMC regularization:"
echo "  PRETRAINED_STUDENT_PATH: ${PRETRAINED_STUDENT_PATH}"
echo "  MODEL_FILE: ${MODEL_FILE}"
echo "  LAMBDA_REG (L2 regularization): ${LAMBDA_REG}"
echo "  BETA_LMC (connectivity loss): ${BETA_LMC}"
echo "  NUM_SAMPLES (interpolation points): ${NUM_SAMPLES}"

CUDA_VISIBLE_DEVICES=0 python MERGETUNE/mergetune/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file MERGETUNE/mergetune/configs/datasets/${DATASET}.yaml \
    --config-file MERGETUNE/mergetune/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT} \
    TRAINER.PROMPTKD.USE_LMC True \
    TRAINER.PROMPTKD.PRETRAINED_STUDENT_PATH ${PRETRAINED_STUDENT_PATH} \
    TRAINER.PROMPTKD.LAMBDA_REG ${LAMBDA_REG} \
    TRAINER.PROMPTKD.BETA_LMC ${BETA_LMC} \
    TRAINER.PROMPTKD.NUM_SAMPLES ${NUM_SAMPLES}

echo "Training completed. Model saved in: ${DIR}"
echo "Used pretrained student model from: ${PRETRAINED_STUDENT_PATH}"
echo "New trained model can be used for future LMC training with path: ${DIR}/VLPromptLearner/"
