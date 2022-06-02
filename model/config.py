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
OUTPUT_DIR = './trained_model/comet-bert/'
SEED = 2022
COMET_FILE = 'xReact'  # COMET relations



# model
BERT_MODEL = 'bert-base-uncased'
COMET_MODEL = './comet-atomic_2020_BART'

# hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 5
EVAL_BATCH_SIZE = 1  # set to 1 for comet model
MAX_SEQ_LENGTH = 64
MAX_COMET_LENGTH = 64
LEARNING_RATE = 5e-5