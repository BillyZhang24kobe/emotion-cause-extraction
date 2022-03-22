from lib2to3.pgen2 import token
from mimetypes import init
from numpy import average
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import numpy as np
from sklearn import metrics
from collections import defaultdict

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from dataset import *

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics.v1 import _precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

# evaluator class
class Evaluator(object):
    def __init__(self, args, model, tokenizers, file_type, label_map):
        self.model = model
        self.tokenizers = tokenizers
        self.file_type = file_type
        self.label_map = label_map

    def span_extraction_metrics(self, args, eval_dataloader, token_map):
        y_true = []
        y_pred = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                outputs = self.model(args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()

            logits_m = logits * input_m

            if args.evaluation_metrics == 'ECSE':
                for i, pred in enumerate(logits_m):
                    temp_1 = []
                    temp_2 = []

                    pred_b_cau_idx = np.where(pred==2)[0].reshape(1,-1)  # 2 -> B-CAU
                    pred_i_cau_idx = np.where(pred==3)[0].reshape(1,-1)  # 3 -> I-CAU
                    pred_cau_idx = np.concatenate((pred_b_cau_idx, pred_i_cau_idx), axis=1)[0]  # cause span indices in y_pred

                    true_b_cau_idx = np.where(label_ids[i]==2)[0].reshape(1,-1)
                    true_i_cau_idx = np.where(label_ids[i]==3)[0].reshape(1,-1)
                    true_cau_idx = np.concatenate((true_b_cau_idx, true_i_cau_idx), axis=1)[0]

                    cau_idx = np.union1d(pred_cau_idx, true_cau_idx)

                    for j in cau_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        temp_1.append(token_map[label_ids[i][j]])
                        temp_2.append(token_map[logits_m[i][j]])

                    y_true.append(temp_1)
                    y_pred.append(temp_2)

            elif args.evaluation_metrics == 'EESE':
                for i, pred in enumerate(logits_m):
                    temp_1 = []
                    temp_2 = []

                    pred_b_emo_idx = np.where(pred==4)[0].reshape(1,-1)  # 4 -> B-EMO
                    pred_i_emo_idx = np.where(pred==5)[0].reshape(1,-1)  # 5 -> I-EMO
                    pred_emo_idx = np.concatenate((pred_b_emo_idx, pred_i_emo_idx), axis=1)[0]  # emotion span indices in y_pred

                    true_b_emo_idx = np.where(label_ids[i]==4)[0].reshape(1,-1)
                    true_i_emo_idx = np.where(label_ids[i]==5)[0].reshape(1,-1)
                    true_emo_idx = np.concatenate((true_b_emo_idx, true_i_emo_idx), axis=1)[0]

                    emo_idx = np.union1d(pred_emo_idx, true_emo_idx)

                    for j in emo_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        temp_1.append(token_map[label_ids[i][j]])
                        temp_2.append(token_map[logits_m[i][j]])

                    y_true.append(temp_1)
                    y_pred.append(temp_2)

            elif args.evaluation_metrics == 'EESE+ECSE':
                for i, pred in enumerate(logits_m):
                    temp_1 = []
                    temp_2 = []

                    pred_b_emo_idx = np.where(pred==4)[0].reshape(1,-1)  # 4 -> B-EMO
                    pred_i_emo_idx = np.where(pred==5)[0].reshape(1,-1)  # 5 -> I-EMO
                    pred_emo_idx = np.concatenate((pred_b_emo_idx, pred_i_emo_idx), axis=1)  # emotion span indices in y_pred
                    pred_b_cau_idx = np.where(pred==2)[0].reshape(1,-1)
                    pred_i_cau_idx = np.where(pred==3)[0].reshape(1,-1)
                    pred_cau_idx = np.concatenate((pred_b_cau_idx, pred_i_cau_idx), axis=1)
                    pred_ec_idx = np.concatenate((pred_emo_idx, pred_cau_idx), axis=1)[0]

                    true_b_emo_idx = np.where(label_ids[i]==4)[0].reshape(1,-1)
                    true_i_emo_idx = np.where(label_ids[i]==5)[0].reshape(1,-1)
                    true_emo_idx = np.concatenate((true_b_emo_idx, true_i_emo_idx), axis=1)
                    true_b_cau_idx = np.where(label_ids[i]==2)[0].reshape(1,-1)
                    true_i_cau_idx = np.where(label_ids[i]==3)[0].reshape(1,-1)
                    true_cau_idx = np.concatenate((true_b_cau_idx, true_i_cau_idx), axis=1)
                    true_ec_idx = np.concatenate((true_emo_idx, true_cau_idx), axis=1)[0]

                    ec_idx = np.union1d(pred_ec_idx, true_ec_idx)

                    for j in ec_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        temp_1.append(token_map[label_ids[i][j]])
                        temp_2.append(token_map[logits_m[i][j]])

                    y_true.append(temp_1)
                    y_pred.append(temp_2)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if self.file_type == 'test':
            report = classification_report(y_true, y_pred,digits=4)
            logger.info("\n%s", report)
            
        return accuracy, precision, recall, f1

    def span_pair_extraction_metrics(self, args, eval_dataloader, token_map):
        def extract_tp_actual_correct(y_true, y_pred, suffix, scheme):
            entities_true = set()  # a set of tuple of tuples
            entities_pred = set()
            emo_start_true, emo_start_pred = None, None
            emo_end_true, emo_end_pred = None, None
            cau_start_true, cau_start_pred = None, None
            cau_end_true, cau_end_pred = None, None
            for type_name, start, end in get_entities(y_true, suffix):
                if type_name == 'EMO':
                    emo_start_true = start
                    emo_end_true = end
                elif type_name == 'CAU':
                    cau_start_true = start
                    cau_end_true = end
                if emo_start_true and emo_end_true and cau_start_true and cau_end_true:
                    # emotion-cause span-pair
                    entities_true.add(((emo_start_true, emo_end_true),(cau_start_true, cau_end_true)))
                    emo_start_true, emo_end_true, cau_start_true, cau_end_true = None, None, None, None
            for type_name, start, end in get_entities(y_pred, suffix):
                if type_name == 'EMO':
                    emo_start_pred = start
                    emo_end_pred = end
                elif type_name == 'CAU':
                    cau_start_pred = start
                    cau_end_pred = end
                if emo_start_pred and emo_end_pred and cau_start_pred and cau_end_pred:
                    entities_pred.add(((emo_start_pred, emo_end_pred),(cau_start_pred, cau_end_pred)))  # ???
                    emo_start_pred, emo_end_pred, cau_start_pred, cau_end_pred =  None, None, None, None

            tp_sum = np.array([len(entities_true & entities_pred)])
            pred_sum = np.array([len(entities_pred)])
            true_sum = np.array([len(entities_true)])

            return pred_sum, tp_sum, true_sum

        y_true = []
        y_pred = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                outputs = self.model(args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()

            logits_m = logits * input_m

            for i, pred in enumerate(logits_m):
                temp_1 = []
                temp_2 = []

                pred_b_emo_idx = np.where(pred==4)[0].reshape(1,-1)  # 4 -> B-EMO
                pred_i_emo_idx = np.where(pred==5)[0].reshape(1,-1)  # 5 -> I-EMO
                pred_emo_idx = np.concatenate((pred_b_emo_idx, pred_i_emo_idx), axis=1)  # emotion span indices in y_pred
                pred_b_cau_idx = np.where(pred==2)[0].reshape(1,-1)
                pred_i_cau_idx = np.where(pred==3)[0].reshape(1,-1)
                pred_cau_idx = np.concatenate((pred_b_cau_idx, pred_i_cau_idx), axis=1)
                pred_ec_idx = np.concatenate((pred_emo_idx, pred_cau_idx), axis=1)[0]

                true_b_emo_idx = np.where(label_ids[i]==4)[0].reshape(1,-1)
                true_i_emo_idx = np.where(label_ids[i]==5)[0].reshape(1,-1)
                true_emo_idx = np.concatenate((true_b_emo_idx, true_i_emo_idx), axis=1)
                true_b_cau_idx = np.where(label_ids[i]==2)[0].reshape(1,-1)
                true_i_cau_idx = np.where(label_ids[i]==3)[0].reshape(1,-1)
                true_cau_idx = np.concatenate((true_b_cau_idx, true_i_cau_idx), axis=1)
                true_ec_idx = np.concatenate((true_emo_idx, true_cau_idx), axis=1)[0]

                ec_idx = np.union1d(pred_ec_idx, true_ec_idx)

                for j in ec_idx:
                    if j == 0:
                        continue
                    # try:
                    if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                        continue
                    temp_1.append(token_map[label_ids[i][j]])
                    temp_2.append(token_map[logits_m[i][j]])

                y_true.append(temp_1)
                y_pred.append(temp_2)

        precision, recall, f_score, _ = _precision_recall_fscore_support(
            y_true, y_pred,
            average='micro',
            warn_for=('precision', 'recall', 'f-score'),
            beta=1.0,
            zero_division='warn',
            suffix=False,
            extract_tp_actual_correct=extract_tp_actual_correct
        )

        return 0, precision, recall, f_score

    def token_level_metrics(self, args, eval_dataloader, token_map):
        y_true_prec = []  # pred focus
        y_pred_prec = []
        y_true_recall = []  # ref focus
        y_pred_recall = []
        y_true = []  # for accuracy
        y_pred = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                outputs = self.model(args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()

            logits_m = logits * input_m

            if args.evaluation_metrics == 'TC':
                for i, pred in enumerate(logits_m):

                    pred_b_cau_idx = np.where(pred==2)[0].reshape(1,-1)  # 2 -> B-CAU
                    pred_i_cau_idx = np.where(pred==3)[0].reshape(1,-1)  # 3 -> I-CAU
                    pred_cau_idx = np.concatenate((pred_b_cau_idx, pred_i_cau_idx), axis=1)[0]  # cause span indices in y_pred
                    for j in pred_cau_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_prec.append(token_map[label_ids[i][j]])
                        y_pred_prec.append(token_map[logits_m[i][j]])

                    true_b_cau_idx = np.where(label_ids[i]==2)[0].reshape(1,-1)
                    true_i_cau_idx = np.where(label_ids[i]==3)[0].reshape(1,-1)
                    true_cau_idx = np.concatenate((true_b_cau_idx, true_i_cau_idx), axis=1)[0]
                    for j in true_cau_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_recall.append(token_map[label_ids[i][j]])
                        y_pred_recall.append(token_map[logits_m[i][j]])

                    cau_idx = np.union1d(pred_cau_idx, true_cau_idx)
                    for j in cau_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true.append(token_map[label_ids[i][j]])
                        y_pred.append(token_map[logits_m[i][j]])

            elif args.evaluation_metrics == 'TE':
                for i, pred in enumerate(logits_m):

                    pred_b_emo_idx = np.where(pred==4)[0].reshape(1,-1)  # 4 -> B-EMO
                    pred_i_emo_idx = np.where(pred==5)[0].reshape(1,-1)  # 5 -> I-EMO
                    pred_emo_idx = np.concatenate((pred_b_emo_idx, pred_i_emo_idx), axis=1)[0]  # emotion span indices in y_pred
                    for j in pred_emo_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_prec.append(token_map[label_ids[i][j]])
                        y_pred_prec.append(token_map[logits_m[i][j]])

                    true_b_emo_idx = np.where(label_ids[i]==4)[0].reshape(1,-1)
                    true_i_emo_idx = np.where(label_ids[i]==5)[0].reshape(1,-1)
                    true_emo_idx = np.concatenate((true_b_emo_idx, true_i_emo_idx), axis=1)[0]
                    for j in true_emo_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_recall.append(token_map[label_ids[i][j]])
                        y_pred_recall.append(token_map[logits_m[i][j]])

                    emo_idx = np.union1d(pred_emo_idx, true_emo_idx)
                    for j in emo_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true.append(token_map[label_ids[i][j]])
                        y_pred.append(token_map[logits_m[i][j]])

            elif args.evaluation_metrics == 'TEC':
                for i, pred in enumerate(logits_m):
                    pred_b_emo_idx = np.where(pred==4)[0].reshape(1,-1)  # 4 -> B-EMO
                    pred_i_emo_idx = np.where(pred==5)[0].reshape(1,-1)  # 5 -> I-EMO
                    pred_emo_idx = np.concatenate((pred_b_emo_idx, pred_i_emo_idx), axis=1)  # emotion span indices in y_pred
                    pred_b_cau_idx = np.where(pred==2)[0].reshape(1,-1)
                    pred_i_cau_idx = np.where(pred==3)[0].reshape(1,-1)
                    pred_cau_idx = np.concatenate((pred_b_cau_idx, pred_i_cau_idx), axis=1)
                    pred_ec_idx = np.concatenate((pred_emo_idx, pred_cau_idx), axis=1)[0]
                    for j in pred_ec_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_prec.append(token_map[label_ids[i][j]])
                        y_pred_prec.append(token_map[logits_m[i][j]])

                    true_b_emo_idx = np.where(label_ids[i]==4)[0].reshape(1,-1)
                    true_i_emo_idx = np.where(label_ids[i]==5)[0].reshape(1,-1)
                    true_emo_idx = np.concatenate((true_b_emo_idx, true_i_emo_idx), axis=1)
                    true_b_cau_idx = np.where(label_ids[i]==2)[0].reshape(1,-1)
                    true_i_cau_idx = np.where(label_ids[i]==3)[0].reshape(1,-1)
                    true_cau_idx = np.concatenate((true_b_cau_idx, true_i_cau_idx), axis=1)
                    true_ec_idx = np.concatenate((true_emo_idx, true_cau_idx), axis=1)[0]
                    for j in true_ec_idx:
                        if j == 0:
                            continue
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true_recall.append(token_map[label_ids[i][j]])
                        y_pred_recall.append(token_map[logits_m[i][j]])

                    ec_idx = np.union1d(pred_ec_idx, true_ec_idx)
                    for j in ec_idx:
                        if j == 0:
                            continue
                        # try:
                        if label_ids[i][j] == 0 or logits_m[i][j] == 0:
                            continue
                        y_true.append(token_map[label_ids[i][j]])
                        y_pred.append(token_map[logits_m[i][j]])

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true_prec, y_pred_prec, average='micro')
        recall = metrics.recall_score(y_true_recall, y_pred_recall, average='micro')
        f1 = (2 * precision * recall) / (precision + recall)

        # if self.file_type == 'test':
        #     report = classification_report(y_true, y_pred,digits=4)
        #     logger.info("\n%s", report)
            
        return accuracy, precision, recall, f1
    
    def evaluate(self, args):
        token_map = self.label_map  # token classification
        # emotion_map = label_map[1]  # emotion classification

        results = {}
        # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, self.tokenizers, file_type=self.file_type)

        # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        #     os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(self.file_type))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.evaluation_metrics in ['ECSE', 'EESE', 'EESE+ECSE']:
            return self.span_extraction_metrics(args, eval_dataloader, token_map)
        elif args.evaluation_metrics in ['TC', 'TE', 'TEC']:
            return self.token_level_metrics(args, eval_dataloader, token_map)
        elif args.evaluation_metrics == 'ECSP':
            return self.span_pair_extraction_metrics(args, eval_dataloader, token_map)