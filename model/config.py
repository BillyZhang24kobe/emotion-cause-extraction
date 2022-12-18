from pytorch_transformers import BertConfig, BertForTokenClassification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'comet-bert': (BartConfig, BartForConditionalGeneration, BartTokenizer)
}

# general
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# data
CONTINUE_DIR = None
OUTPUT_DIR = './trained_model/bert-doc-gpt3/'
SEED = 2022
COMET_FILE = 'xReact'  # COMET relations
GLUCOSE_FILE = 'dim6'
GPT3_SHOT_TYPE = 'TRS-4'  # GPT-3 generated explanations with different shot type

# model
BERT_MODEL = 'bert-base-uncased'
COMET_MODEL = './comet-atomic_2020_BART'

# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 32  # set to 1 for comet model
MAX_SEQ_LENGTH = 128
MAX_COMET_LENGTH = 128
LEARNING_RATE = 5e-5

# python main.py --model_class bert-gpt3 --data_dir ../data --do_eval --evaluation_metrics EESE --task_name eca-comet

# python main.py --model_class bert-gpt3 --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca

# nohup python main.py --model_class bert-gpt3 --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca > bert_gpt3_32_10_256_LR_ECSE.txt &