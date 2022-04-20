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

# from IPython import embed

import logging

logger = logging.getLogger(__name__)

def combined_pred_true(logits, label_ids, input_m):
    """
    Combine the clause logits to form document logits
    """
    # output: batch_size x max_seq_len
    logits_concat = np.array([[]])
    labels_concat = np.array([[]])
    for lgt, lbs in zip(logits, label_ids):
        pred = lgt[np.nonzero(lbs)]
        true = lbs[np.nonzero(lbs)]
        
        logits_concat = np.append(logits_concat, [pred], axis=1)
        labels_concat = np.append(labels_concat, [true], axis=1)

    # print(logits_concat)
    # print(logits_concat.shape)
    # print(labels_concat)
    # print(labels_concat.shape)


    while logits_concat.shape[1] < input_m.shape[1]:
        logits_concat = np.append(logits_concat, [[0]], axis=1)
        labels_concat = np.append(labels_concat, [[0]], axis=1)

    return logits_concat, labels_concat


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
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                if 'comet' in args.model_class:
                    input_ids = input_ids.squeeze(0)
                    input_mask = input_mask.squeeze(0)
                    segment_ids = segment_ids.squeeze(0)
                    label_ids = label_ids.squeeze(0)
                    valid_ids = valid_ids.squeeze(0)
                    l_mask = l_mask.squeeze(0)

                outputs = self.model(args, args.device, input_ids, segment_ids, input_mask, valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]
                # print(logits.shape)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()
            if 'comet' in args.model_class:
                logits_m, label_ids = combined_pred_true(logits, label_ids, input_m)
            else:
                logits_m = logits * input_m  # batch_size x max_seq_len


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

                    # embed()
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

        if args.print_report:
            report = classification_report(y_true, y_pred,digits=4)
            logger.info("\n%s", report)

        if args.print_prediction:
            with open('pred_output.txt', 'w') as file:
                for pred in y_pred:
                    file.write(' '.join(pred) + '\n')

            with open('true_output.txt', 'w') as file:
                for true in y_true:
                    file.write(' '.join(true) + '\n')
            
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
                if 'comet' in args.model_class:
                    input_ids = input_ids.squeeze(0)
                    input_mask = input_mask.squeeze(0)
                    segment_ids = segment_ids.squeeze(0)
                    label_ids = label_ids.squeeze(0)
                    valid_ids = valid_ids.squeeze(0)
                    l_mask = l_mask.squeeze(0)

                outputs = self.model(args, args.device, input_ids, segment_ids, input_mask, valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()
            if 'comet' in args.model_class:
                logits_m, label_ids = combined_pred_true(logits, label_ids, input_m)
            else:
                logits_m = logits * input_m  # batch_size x max_seq_len

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

        if args.print_report:
            report = classification_report(y_true, y_pred,digits=4)
            logger.info("\n%s", report)

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
                if 'comet' in args.model_class:
                    input_ids = input_ids.squeeze(0)
                    input_mask = input_mask.squeeze(0)
                    segment_ids = segment_ids.squeeze(0)
                    label_ids = label_ids.squeeze(0)
                    valid_ids = valid_ids.squeeze(0)
                    l_mask = l_mask.squeeze(0)
                outputs = self.model(args, args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                logits = outputs #[:2]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_m = input_mask.to('cpu').numpy()
            if 'comet' in args.model_class:
                logits_m, label_ids = combined_pred_true(logits, label_ids, input_m)
            else:
                logits_m = logits * input_m  # batch_size x max_seq_len

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

        if args.print_report:
            report = classification_report(y_true, y_pred,digits=4)
            logger.info("\n%s", report)
            
        return accuracy, precision, recall, f1
    
    def emotion_prediction(self, args, eval_dataloader, emotion_map):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                outputs = self.model(args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

                # embed()
                # print(outputs)

                n_correct += (torch.argmax(outputs, -1) == label_ids.view(-1)).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = label_ids
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, label_ids), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

        accuracy = n_correct / n_total
        precision = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2, 3, 4, 5], average='micro')
        recall = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2, 3, 4, 5], average='micro')
        f1 = (2 * precision * recall) / (precision + recall)

        if args.print_report:
            report = metrics.classification_report(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu())
            logger.info("\n%s", report)

        return accuracy, precision, recall, f1

    
    def evaluate(self, args):
        args.eval_batch_size = args.per_gpu_eval_batch_size

        if 'comet' in args.model_class:
            eval_dataset = CometEvalDataset(args, self.file_type, self.tokenizers)
        else:
            eval_dataset = load_and_cache_examples(args, self.tokenizers, file_type=self.file_type)

            # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(self.file_type))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.evaluation_metrics == 'emotion_clf':
            return self.emotion_prediction(args, eval_dataloader, self.label_map)

        if args.evaluation_metrics in ['ECSE', 'EESE', 'EESE+ECSE']:
            return self.span_extraction_metrics(args, eval_dataloader, self.label_map)
        elif args.evaluation_metrics in ['TC', 'TE', 'TEC']:
            return self.token_level_metrics(args, eval_dataloader, self.label_map)
        elif args.evaluation_metrics == 'ECSP':
            return self.span_pair_extraction_metrics(args, eval_dataloader, self.label_map)