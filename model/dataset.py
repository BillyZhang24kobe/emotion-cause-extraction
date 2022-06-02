import os
import csv
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)

import logging

logger = logging.getLogger(__name__)
from IPython import embed

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


# def read_comet_file(filename):
#     tsv_file = open(filename)
#     read_tsv = csv.reader(tsv_file, delimiter="\t")
#     data = []
#     for i, line in enumerate(read_tsv):
#         if i == 0: continue
#         clause = line[0]
#         document = line[1]
#         clause_label = line[2]
#         emotion_label = line[3]
#         doc_id = line[4]

#         # x_output = line[5]
#         # data.append((clause, document, clause_label, emotion_label, doc_id, x_output))
        
#         data.append((clause, document, clause_label, emotion_label, doc_id))
    
#     return data


# def create_examples(lines, set_type):
#         examples = []
#         # for i,(clause, document, clause_label, emotion_label, doc_id, x_response) in enumerate(lines):
#         for i,(clause, document, clause_label, emotion_label, doc_id) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             # text_a = (clause, document, doc_id, x_response)
#             text_a = (clause, document, doc_id)
#             text_b = None
#             labels = (clause_label, emotion_label)
#             examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=labels))
#         return examples


# def get_labels():
#     token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
#     # token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO"]
#     emotion_labels = ['fear', 'surprise', 'disgust', 'sadness', 'anger', 'happiness']
#     return token_labels, emotion_labels


class ClauseDataset(TensorDataset):

    def __init__(self, args, eval_file_type, tokenizers):
        self.args = args
        self.eval_file_type = eval_file_type
        self.label_tuple = ClauseDataset.get_labels()
        self.tokenizers = tokenizers
        if self.args.task_name == 'eca-clause':
            if self.eval_file_type == 'train':
                self.data = ClauseDataset.read_clause_file(os.path.join(args.data_dir, "clause-train-pair.tsv"))
            elif self.eval_file_type == 'dev':
                self.data = ClauseDataset.read_clause_file(os.path.join(args.data_dir, "clause-dev-pair.tsv"))
            else:
                self.data = ClauseDataset.read_clause_file(os.path.join(args.data_dir, "clause-test-pair.tsv"))
        elif self.args.task_name == 'eca-comet':
            if self.eval_file_type == 'train':
                self.data = ClauseDataset.read_comet_file(os.path.join(args.data_dir, "comet-train-pair-{}.tsv".format(args.comet_file)))
            elif self.eval_file_type == 'dev':
                self.data = ClauseDataset.read_comet_file(os.path.join(args.data_dir, "comet-dev-pair-{}.tsv".format(args.comet_file)))
            else:
                self.data = ClauseDataset.read_comet_file(os.path.join(args.data_dir, "comet-test-pair-{}.tsv".format(args.comet_file)))
        # self.data = read_comet_file(os.path.join(args.data_dir, "comet-dev-pair.tsv")) if eval_file_type == 'dev' else read_comet_file(os.path.join(args.data_dir, "comet-test-pair.tsv"))  # a list of tuples
        # self.eval_data = read_comet_file(os.path.join(args.data_dir, "comet-train-pair-sample.tsv"))
        self.idx_to_doc_id = {}
        idx = -1  # batch index
        doc_id_prev = -1
        for data in self.data:
            if data[4] != doc_id_prev:
                idx += 1
                self.idx_to_doc_id[idx] = data[4]
                doc_id_prev = data[4]
        
    def __len__(self):
        return len(self.idx_to_doc_id.keys())

    def __getitem__(self, idx):
        doc_id = self.idx_to_doc_id[idx]
        lines = []
        for data in self.data:
            if data[4] == doc_id:
                lines.append(data)
            elif len(lines) != 0:
                break
        if self.args.task_name == 'eca-clause': 
            examples = ClauseDataset.create_clause_examples(lines, self.eval_file_type)
            features = convert_clause_examples_to_features(self.args, examples, self.label_tuple, self.args.max_seq_length, self.tokenizers)
        elif self.args.task_name == 'eca-comet':
            examples = ClauseDataset.create_comet_examples(lines, self.eval_file_type)
            features = convert_comet_examples_to_features(self.args, examples, self.label_tuple, self.args.max_seq_length, self.tokenizers)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

        dataset = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
        
        return dataset

    @staticmethod
    def read_clause_file(filename):
        tsv_file = open(filename)
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data = []
        for i, line in enumerate(read_tsv):
            if i == 0: continue
            clause = line[0]
            document = line[1]
            clause_label = line[2]
            emotion_label = line[3]
            doc_id = line[4]

            data.append((clause, document, clause_label, emotion_label, doc_id))
        
        return data

    @staticmethod
    def read_comet_file(filename):
        tsv_file = open(filename)
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data = []
        for i, line in enumerate(read_tsv):
            if i == 0: continue
            clause = line[0]
            document = line[1]
            clause_label = line[2]
            emotion_label = line[3]
            doc_id = line[4]
            x_output = line[5]
            data.append((clause, document, clause_label, emotion_label, doc_id, x_output))
        
        return data

    @staticmethod
    def create_clause_examples(lines, set_type):
        examples = []
        for i,(clause, document, clause_label, emotion_label, doc_id) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = (clause, document, doc_id)
            text_b = None
            labels = (clause_label, emotion_label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=labels))
        return examples

    @staticmethod
    def create_comet_examples(lines, set_type):
        examples = []
        for i,(clause, document, clause_label, emotion_label, doc_id, x_response) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = (clause, document, doc_id, x_response)
            text_b = None
            labels = (clause_label, emotion_label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=labels))
        return examples

    @staticmethod
    def get_labels():
        token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
        # token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO"]
        emotion_labels = ['fear', 'surprise', 'disgust', 'sadness', 'anger', 'happiness']
        return token_labels, emotion_labels


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
        # self.doc_id = doc_id


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

    @classmethod
    def _read_comet_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return read_comet_file(input_file)


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
        # token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO"]
        emotion_labels = ['fear', 'surprise', 'disgust', 'sadness', 'anger', 'happiness']
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


class COMETProcessor(DataProcessor):
    """Processor for eca data set"""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_comet_tsv(os.path.join(data_dir, "comet-train-pair.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_comet_tsv(os.path.join(data_dir, "comet-dev-pair.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_comet_tsv(os.path.join(data_dir, "comet-test-pair.tsv")), "test")
        
    def get_labels(self):
        token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
        # token_labels = ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO"]
        emotion_labels = ['fear', 'surprise', 'disgust', 'sadness', 'anger', 'happiness']
        return token_labels, emotion_labels
    
    def _create_examples(self,lines,set_type):
        examples = []
        # for i, (clause, document, clause_label, emotion_label, doc_id, x_output) in enumerate(lines):
        for i, (clause, document, clause_label, emotion_label, doc_id) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text_a = (clause, document, doc_id, x_output)
            text_a = (clause, document, doc_id)
            text_b = None
            labels = (clause_label, emotion_label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=labels))
        return examples


def convert_examples_to_features(args, examples, label_tuple, max_seq_length, tokenizers):
    """Loads a data file into a list of `InputBatch`s."""
    token_labels, emotion_labels = label_tuple

    token_label_map = {label : i for i, label in enumerate(token_labels, 1)}
    emotion_label_map = {label : i for i, label in enumerate(emotion_labels, 1)}

    bert_tokenizer = tokenizers[0]
    
    features = []
    for (ex_index,example) in enumerate(examples):
        # text_a = (clause, document, doc_id)
        document = example.text_a.split(' ')
        token_label, emotion_label = example.label
        bert_tokens = []
        # comet_tokens = []
        labels = []
        valid = []  # TODO: what is this?
        label_mask = []
        for i, word in enumerate(document):
            bert_token = bert_tokenizer.tokenize(word.lower())
            bert_tokens.extend(bert_token)
            label_1 = token_label.split()[i]
            for m in range(len(bert_token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        # embed()
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
        
        all_input_ids = bert_input_ids

        # embed()
        features.append(
                InputFeatures(input_ids=all_input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


def convert_examples_to_emotion_features(args, examples, label_tuple, max_seq_length, tokenizers):
    """Loads a data file into a list of `InputBatch`s."""
    token_labels, emotion_labels = label_tuple

    # token_label_map = {label : i for i, label in enumerate(token_labels, 1)}
    emotion_label_map = {label : i for i, label in enumerate(emotion_labels)}

    bert_tokenizer = tokenizers[0]
    
    features = []
    for (ex_index,example) in enumerate(examples):
        document = example.text_a.split(' ')
        token_label, emotion_label = example.label

        bert_input_ids = bert_tokenizer.encode(document, add_special_tokens=True, max_length=max_seq_length, truncation=True, padding='max_length')

        # Create attention mask
        attention_mask = []
        ## Create a mask of 1 for all input tokens and 0 for all padding tokens
        attention_mask = [float(i>0) for i in bert_input_ids]

        label_ids = []
        label_ids.append(emotion_label_map[emotion_label])
        
        segment_ids = [0] * len(bert_input_ids)
        valid = [0] * len(bert_input_ids)
        label_mask = [0] * len(bert_input_ids)

        # bert_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_ntokens)
        # input_mask = [1] * len(bert_input_ids)
        # label_mask = [1] * len(bert_input_ids)
        # while len(bert_input_ids) < max_seq_length:  # padding
        #     bert_input_ids.append(0)
        #     input_mask.append(0)
        #     segment_ids.append(0)
        #     valid.append(1)
        #     label_mask.append(0)

        assert len(bert_input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == 1
        assert len(valid) == max_seq_length
    
        features.append(
                InputFeatures(input_ids=bert_input_ids,
                              input_mask=attention_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


##### version 2 COMET features (bert-clause) ######
def convert_clause_examples_to_features(args, examples, label_tuple, max_seq_length, tokenizers):
    """Loads a data file into a list of `InputBatch`s."""
    token_labels, emotion_labels = label_tuple

    token_label_map = {label : i for i, label in enumerate(token_labels, 1)}
    emotion_label_map = {label : i for i, label in enumerate(emotion_labels, 1)}

    bert_tokenizer = tokenizers[0]
    
    features = []
    for (ex_index, example) in enumerate(examples):
        clause = example.text_a[0].strip().split(' ')
        # document = re.sub('\s+',' ', example.text_a[1]).strip().split(' ')
        token_label, emotion_label = example.label
        bert_tokens = []
        labels = []
        valid = []  # TODO: what is this?
        label_mask = []
        for i, word in enumerate(clause):
            bert_token = bert_tokenizer.tokenize(word.lower())
            bert_tokens.extend(bert_token)
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

        bert_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_ntokens)
        input_mask = [1] * len(bert_input_ids)
        label_mask = [1] * len(label_ids)
        while len(bert_input_ids) < max_seq_length:  # padding for bert
            bert_input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        
        while len(label_ids) < args.max_comet_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(bert_input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in bert_tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in bert_input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        # combine bert and comet input ids
        # if 'comet' in args.model_class:
        # embed()
        all_input_ids = bert_input_ids # 1 x (bert_max_seq_len + comet_max_len)

        # if 'comet' in args.model_class:
        assert len(all_input_ids) == (args.max_seq_length)

        # embed()
        features.append(
                InputFeatures(input_ids=all_input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


##### BERT-Clause + COMET Relations joint input to Bert #####
def convert_comet_examples_to_features(args, examples, label_tuple, max_seq_length, tokenizers):
    """Loads a data file into a list of `InputBatch`s."""
    token_labels, emotion_labels = label_tuple

    token_label_map = {label : i for i, label in enumerate(token_labels, 1)}
    emotion_label_map = {label : i for i, label in enumerate(emotion_labels, 1)}

    bert_tokenizer = tokenizers[0]
    # comet_tokenizer = tokenizers[1]
    
    features = []
    for (ex_index, example) in enumerate(examples):
        clause = example.text_a[0].strip().split(' ')
        xReact_response = example.text_a[3].strip().split(' ')
        # document = re.sub('\s+',' ', example.text_a[1]).strip().split(' ')
        token_label, emotion_label = example.label
        bert_tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(clause):
            bert_token = bert_tokenizer.tokenize(word.lower())
            bert_tokens.extend(bert_token)
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
        segment_ids = []
        label_ids = []

        # encoding clause input
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

        # encoding xReact_output as input to BERT
        for i, word in enumerate(xReact_response):
            bert_token = bert_tokenizer.tokenize(word.lower())
            for i, b_token in enumerate(bert_token):
                bert_ntokens.append(b_token)
                segment_ids.append(1)
                valid.append(1)
                label_mask.append(0)
                label_ids.append(0)
        
        bert_ntokens.append("[SEP]")
        segment_ids.append(1)
        valid.append(1)
        label_mask.append(0)
        label_ids.append(0)

        bert_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_ntokens)
        input_mask = [1] * len(bert_input_ids)
        # label_mask = [1] * len(label_ids)  # TODO: maybe this causes problems?
        while len(bert_input_ids) < max_seq_length:  # padding for bert
            bert_input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        if len(bert_input_ids) > max_seq_length:
            bert_input_ids = bert_input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
            segment_ids = segment_ids[:max_seq_length]
            label_ids = label_ids[:max_seq_length]
            valid = valid[:max_seq_length]
            label_mask = label_mask[:max_seq_length]

        assert len(bert_input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in bert_tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in bert_input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        # combine bert and comet input ids
        # if 'comet' in args.model_class:
        # embed()
        all_input_ids = bert_input_ids # 1 x (bert_max_seq_len)

        # if 'comet' in args.model_class:
        assert len(all_input_ids) == args.max_seq_length

        # embed()
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

    if 'comet' in args.model_class:
        processor = COMETProcessor()
    else:
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

        if file_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif file_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        if 'comet' in args.model_class:
            features = convert_comet_examples_to_features(args, examples, label_tuple, args.max_seq_length, tokenizers)
        elif 'emotion' in args.model_class:
            features = convert_examples_to_emotion_features(args, examples, label_tuple, args.max_seq_length, tokenizers)
        else:
            features = convert_examples_to_features(args, examples, label_tuple, args.max_seq_length, tokenizers)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    # embed()
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,all_lmask_ids)

    return dataset