from lib2to3.pgen2 import token
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import config
from utils import *
from predict import Evaluator

import logging

logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizers, label_map):
    """
    Train the model
    """
    tb_writer = SummaryWriter()
    # args.save_steps = config.SAVE_STEPS
    args.num_train_epochs = config.NUM_EPOCHS
    args.output_dir = config.OUTPUT_DIR
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = config.BATCH_SIZE
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_val_f1 = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
            # inputs = {'input_ids':      batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2],
            #         #   'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
            #           'labels':         batch[3]}
            outputs = model(args, args.device, input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            loss = outputs

            # if args.n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # tb_writer.add_scalar('eval_{}'.format('f1'), val_f1, global_step)
                # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                logging_loss = tr_loss
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        file_type = 'dev'
        evaluator = Evaluator(args, model, tokenizers, file_type, label_map)
        val_acc, val_precision, val_recall, val_f1 = evaluator.evaluate(args)  # 
        logger.info('> '+ args.evaluation_metrics + 'val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
            # Save model checkpoint
            path_dir = os.path.join(args.output_dir, 'checkpoint_val_f1_{}'.format(round(val_f1, 4))+args.evaluation_metrics)
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            model.save_pretrained(path_dir)
            bert_tokenizer = tokenizers[0]
            bert_tokenizer.save_pretrained(path_dir)
            if 'comet' in args.model_class:
                comet_tokenizer = tokenizers[1]
                comet_tokenizer.save_pretrained(path_dir)
            torch.save(args, os.path.join(path_dir, 'training_args.bin'))
            logger.info(">>Saving model checkpoint to %s", path_dir)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(path)
            

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)
            

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    return global_step, tr_loss / global_step, path_dir