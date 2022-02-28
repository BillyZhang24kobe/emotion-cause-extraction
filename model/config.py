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
OUTPUT_DIR = 'eca-model'
SEED = 1

# model
BERT_MODEL = 'bert-base-uncased'

# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
SAVE_STEPS = 750
MAX_LENGTH = 512