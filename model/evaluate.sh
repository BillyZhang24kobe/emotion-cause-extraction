#!/bin/bash

# training
# LOG_DIR='./logs/model_checkpoints/bect-Causes-xReason/'
# LOG_DIR='./logs/model_checkpoints/gpt3-TRS-4/'
LOG_DIR='./logs/model_checkpoints/bect-doc/'
OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bect-doc/checkpoint_bert_val_f1_0.2177_ECSE_3872'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/comet-bert/bect-Causes-xReason/'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bect-doc'
COMET_FILE='dim2-spec'
# COMET_FILE='Causes-xReason'
GPT3_SHOT_TYPE='TRS-4'
# eval
EVAL_METRICS='ECSE'
TASK_NAME='eca-comet'

# data
# DATA_DIR='./GPT3/data'
DATA_DIR='../data'

# model
BERT_MODEL='bert-base-uncased'
COMET_MODEL='./COMET/comet-atomic_2020_BART'
MODEL_CLASS='bert'
# MODEL_CLASS='bert'

#hyperparameters
BATCH_SIZE=32
NUM_EPOCHS=10
EVAL_BATCH_SIZE=32  # set to 1 for comet model
MAX_SEQ_LENGTH=64
MAX_COMET_LENGTH=64
LEARNING_RATE=5e-5

DEVICE=2

SEED=3872

python main.py --do_eval --model_class ${MODEL_CLASS} --data_dir ${DATA_DIR} --evaluation_metrics ${EVAL_METRICS} --task_name eca-comet --output_dir ${OUTPUT_DIR} \
--task_name ${TASK_NAME} --comet_file ${COMET_FILE} --bert_model ${BERT_MODEL} \
--max_seq_length ${MAX_SEQ_LENGTH} --max_comet_seq_length ${MAX_COMET_LENGTH} \
--per_gpu_train_batch_size ${BATCH_SIZE} --per_gpu_eval_batch_size ${EVAL_BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} --num_train_epochs ${NUM_EPOCHS} --seed ${SEED} --gpt3_shot_type ${GPT3_SHOT_TYPE} --device ${DEVICE} --overwrite_output_dir
