#!/bin/bash

# training
# LOG_DIR='./logs/model_checkpoints/bect-Causes-xReason/'
# LOG_DIR='./logs/model_checkpoints/gpt3-TRS-4/'
LOG_DIR='./logs/model_checkpoints/bect-doc/'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bert-doc-gpt3/TRS-4/'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/comet-bert/bect-Causes-xReason/'
OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bect-doc'
COMET_FILE='dim2-spec'
# COMET_FILE='Causes-xReason'
GPT3_SHOT_TYPE='TRS-4'
# eval
EVAL_METRICS='ECSP'
TASK_NAME='eca-comet'

# data
# DATA_DIR='./GPT3/data'
DATA_DIR='../data'

# model
BERT_MODEL='bert-base-uncased'
COMET_MODEL='./COMET/comet-atomic_2020_BART'
# MODEL_CLASS='bert-gpt3'
MODEL_CLASS='bert'

#hyperparameters
BATCH_SIZE=32
NUM_EPOCHS=10
EVAL_BATCH_SIZE=32  # set to 1 for comet model
MAX_SEQ_LENGTH=64
MAX_COMET_LENGTH=64
LEARNING_RATE=5e-5

SEED_LIST=(3872 9349 2050 6985 574)
# SEED_LIST=(6985 574)
REPEAT_LIST=(0)
DEVICE=2

for seed in "${SEED_LIST[@]}"
do
  for repeat in "${REPEAT_LIST[@]}"
  do
    python main.py --model_class ${MODEL_CLASS} --data_dir ${DATA_DIR} --do_train \
    --evaluation_metrics ${EVAL_METRICS} --output_dir ${OUTPUT_DIR} --log_dir ${LOG_DIR} \
    --repeat ${repeat} --task_name ${TASK_NAME} --comet_file ${COMET_FILE} --bert_model ${BERT_MODEL} \
    --max_seq_length ${MAX_SEQ_LENGTH} --max_comet_seq_length ${MAX_COMET_LENGTH} \
    --per_gpu_train_batch_size ${BATCH_SIZE} --per_gpu_eval_batch_size ${EVAL_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} --num_train_epochs ${NUM_EPOCHS} --overwrite_output_dir \
    --seed ${seed} --gpt3_shot_type ${GPT3_SHOT_TYPE} --device ${DEVICE}
  done
done
