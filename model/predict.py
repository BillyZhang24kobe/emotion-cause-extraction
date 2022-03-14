import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from dataset import *

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

# TODO: evaluator class (TokenLevel, OverLap)

def evaluate(args, model, tokenizer, file_type, label_map):
    token_map = label_map  # token classification
    # emotion_map = label_map[1]  # emotion classification

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
 
    results = {}
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    eval_dataset = load_and_cache_examples(args, tokenizer, file_type=file_type)

    # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(file_type))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # eval_loss = 0.0
    # nb_eval_steps = 0
    # preds = None
    # out_label_ids = None
    y_true = []
    y_pred = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
            outputs = model(args.device, input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
            logits = outputs #[:2]

        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_m = input_mask.to('cpu').numpy()

        logits_m = logits * input_m

        # token-level-cause metrics
        if args.evaluation_metrics == 'TLC':
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

        elif args.evaluation_metrics == 'TLE':
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

        elif args.evaluation_metrics == 'TLEC':
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

        # for i, label in enumerate(label_ids):
        #     temp_1 = []
        #     temp_2 = []
        #     for j,m in enumerate(label):
        #         if j == 0:
        #             continue
        #         elif label_ids[i][j] == len(token_map):
        #             y_true.append(temp_1)
        #             y_pred.append(temp_2)
        #             break
        #         else:
        #             temp_1.append(token_map[label_ids[i][j]])
        #             temp_2.append(token_map[logits[i][j]])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if file_type == 'test':
        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n%s", report)
        
    return accuracy, precision, recall, f1
    # report = classification_report(y_true, y_pred,digits=4)
    # logger.info("\n%s", report)
    # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     logger.info("\n%s", report)
    #     writer.write(report)