from pytorch_transformers import BertConfig, BertForTokenClassification, BertTokenizer, BertForSequenceClassification
import torch
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'comet-bert': (BartConfig, BartForConditionalGeneration, BartTokenizer)
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

# general
# is_cuda = False
# DEVICE = torch.device("cuda" if is_cuda else "cpu")

# data
OUTPUT_DIR = './trained_model/comet-bert/'
SEED = 2022


# model
BERT_MODEL = 'bert-base-uncased'
COMET_MODEL = './comet-atomic_2020_BART'

# hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 5e-5