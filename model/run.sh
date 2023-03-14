#!/bin/bash

# training
# LOG_DIR='./logs/model_checkpoints/bect-Causes-xReason/'
# LOG_DIR='./logs/model_checkpoints/gpt3-TRS-4-ghazi/fold4/'
# LOG_DIR='./logs/model_checkpoints/bect-ghazi-xEffect-fold4/'
# LOG_DIR='./logs/model_checkpoints/bect-gne-Causes/MaxLen64_Epochs10_Batch16/'
LOG_DIR='./logs/model_checkpoints/bect-gne-xEffect/MaxLen64_Epochs10_Batch16/'
# LOG_DIR='./logs/model_checkpoints/bect-gne/MaxLen32_Epochs20/'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bert-doc-gpt3/TRS-4/'
# OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/comet-bert/bect-Causes-xReason/'
CONTINUE_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bect-gne-xEffect/MaxLen64_Epochs10_Batch16/checkpoint_bert-doc-com_xEffect_val_f1_0.2683_ECSE'
OUTPUT_DIR='/mnt/swordfish-pool2/xz2995/trained_model/bect-gne-xEffect/MaxLen64_Epochs10_Batch16'
COMET_FILE='xEffect'
# COMET_FILE='Causes-xReason'
GPT3_SHOT_TYPE='TRS-4'
# eval
EVAL_METRICS='ECSE'
TASK_NAME='eca-comet'

# data
# DATA_DIR='./GPT3/data/ghazi/fold4'
# DATA_DIR='../data'
# DATA_DIR='../data/ghazi/fold4'
DATA_DIR='./COMET/data/goodnewseveryone/'
# DATA_DIR='../data/goodnewseveryone-v1.0/data'

# model
BERT_MODEL='bert-base-uncased'
COMET_MODEL='./COMET/comet-atomic_2020_BART'
MODEL_CLASS='bert-gpt3'
# MODEL_CLASS='bert-gne'
# MODEL_CLASS='bert-doc-com'
# MODEL_CLASS='bert-comet'

#hyperparameters
BATCH_SIZE=16
NUM_EPOCHS=5
EVAL_BATCH_SIZE=16  # set to 1 for comet model
MAX_SEQ_LENGTH=64
MAX_COMET_LENGTH=64
LEARNING_RATE=1e-4

# SEED_LIST=(3872 9349 2050 6985 574)
SEED_LIST=(6985)
REPEAT_LIST=(0)
DEVICE=0

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
    --seed ${seed} --gpt3_shot_type ${GPT3_SHOT_TYPE} --device ${DEVICE} \
    --continue_dir ${CONTINUE_DIR}
  done
done
