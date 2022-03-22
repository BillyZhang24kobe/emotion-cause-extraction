import os
import csv

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import logging

logger = logging.getLogger(__name__)
from IPython import embed


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    data = []
    for i, line in enumerate(read_tsv):
        if i == 0: continue
        document = line[0]
        doc_label = line[1]
        emotion_label = line[2]
        
        data.append((document, doc_label, emotion_label))
    
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class ECAProcessor(DataProcessor):
    """Processor for eca data set"""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eca-train-cleaned.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eca-dev-cleaned.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eca-test-cleaned.tsv")), "test")
        
    def get_labels(self):
        token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
        emotion_labels = ['Fear', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Happiness']
        return token_labels, emotion_labels
    
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(document, doc_label, emotion_label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = document
            text_b = None
            labels = (doc_label, emotion_label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=labels))
        return examples


def convert_examples_to_features(args, examples, label_tuple, max_seq_length, tokenizers):
    """Loads a data file into a list of `InputBatch`s."""
    token_labels, emotion_labels = label_tuple

    token_label_map = {label : i for i, label in enumerate(token_labels, 1)}
    emotion_label_map = {label : i for i, label in enumerate(emotion_labels, 1)}

    bert_tokenizer = tokenizers[0]
    if 'comet' in args.model_class:
        comet_tokenizer = tokenizers[1]
    
    features = []
    for (ex_index,example) in enumerate(examples):
        document = example.text_a.split(' ')
        token_label, emotion_label = example.label
        bert_tokens = []
        # comet_tokens = []
        labels = []
        valid = []  # TODO: what is this?
        label_mask = []
        for i, word in enumerate(document):
            bert_token = bert_tokenizer.tokenize(word)
            bert_tokens.extend(bert_token)
            # if 'comet' in args.model_class:
            #     comet_token = comet_tokenizer.tokenize(word)
            #     comet_tokens.extend(comet_token)
            label_1 = token_label.split()[i]
            for m in range(len(bert_token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(bert_tokens) >= max_seq_length - 1:
            bert_tokens = bert_tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        bert_ntokens = []
        # comet_ntokens = []
        segment_ids = []
        label_ids = []

        # preprocessing for bert
        bert_ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(token_label_map["[CLS]"])
        for i, token in enumerate(bert_tokens):
            bert_ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(token_label_map[labels[i]])
        bert_ntokens.append("[SEP]")  # append SEP to the end
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(token_label_map["[SEP]"])

        # preprocessing for comet
        # for i, token in enumerate(comet_tokens):
        #     comet_ntokens.append(token)
        # comet_ntokens.append("xEffect")
        # comet_ntokens.append("[GEN]")
        query = "{} {} [GEN]".format(example.text_a, 'xEffect')
        tokenized_sequence = comet_tokenizer.tokenize(query)
        comet_input_ids = comet_tokenizer.encode(tokenized_sequence, truncation=True, padding="max_length")

        bert_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_ntokens)
        input_mask = [1] * len(bert_input_ids)
        label_mask = [1] * len(label_ids)
        while len(bert_input_ids) < max_seq_length:  # padding
            bert_input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(bert_input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in bert_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in bert_input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        # combine bert and comet input ids
        all_input_ids = bert_input_ids + comet_input_ids  # 1 x (bert_max_seq_len + comet_max_len)

        assert len(all_input_ids) == max_seq_length + 1024

        features.append(
                InputFeatures(input_ids=all_input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


def load_and_cache_examples(args, tokenizers, file_type='train'):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = ECAProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        file_type,
        list(filter(None, args.model_class.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_tuple = processor.get_labels()
        # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        #     # HACK(label indices are swapped in RoBERTa pretrained model)
        #     label_list[1], label_list[2] = label_list[2], label_list[1]
        if file_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif file_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        
        features = convert_examples_to_features(args, examples, label_tuple, args.max_seq_length, tokenizers)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset 
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,all_lmask_ids)

    #dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset