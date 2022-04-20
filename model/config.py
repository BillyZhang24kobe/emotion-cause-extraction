from pytorch_transformers import BertConfig, BertForTokenClassification, BertTokenizer, BertForSequenceClassification
import torch
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'comet-bert': (BartConfig, BartForConditionalGeneration, BartTokenizer)
}

# general
# is_cuda = False
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# data
OUTPUT_DIR = './trained_model/comet-bert/'
SEED = 2022


# model
BERT_MODEL = 'bert-base-uncased'
COMET_MODEL = './comet-atomic_2020_BART'

# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
EVAL_BATCH_SIZE = 1  # set to 1 for comet model
MAX_SEQ_LENGTH = 128
MAX_COMET_LENGTH = 128
LEARNING_RATE = 5e-5