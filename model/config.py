from pytorch_transformers import BertConfig, BertForTokenClassification, BertTokenizer, BertForSequenceClassification
import torch

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

# general
# is_cuda = False
# DEVICE = torch.device("cuda" if is_cuda else "cpu")

# data
OUTPUT_DIR = './trained_model/ect_bert/checkpoint_val_f1_0.4157TLEC'
SEED = 2022


# model
BERT_MODEL = 'bert-base-uncased'

# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 5e-5