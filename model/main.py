from config import *

import argparse
from utils import *
from dataset import *
from train import *
from predict import *
from model import *

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.") 
    parser.add_argument("--evaluation_metrics", default=None, type=str, required=True, help="Whether to evaluate using token-level-cause metric.")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    
    
    ## Other parameters
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")   
    parser.add_argument("--task_name", default="eca", type=str,
                        help="The name of the task to train")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model or name")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    # parser.add_argument("--no_cuda", action='store_true',
    #                     help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    args.output_dir = OUTPUT_DIR
    args.model_name_or_path = BERT_MODEL
    args.max_seq_length = MAX_SEQ_LENGTH
    args.per_gpu_train_batch_size = config.BATCH_SIZE
    args.per_gpu_eval_batch_size = config.BATCH_SIZE
    args.learning_rate = config.LEARNING_RATE

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    processor = ECAProcessor()
    label_list = processor.get_labels()[0]  # [0] token_labels
    label_map = {i : label for i, label in enumerate(label_list,1)}  # token_labels
    num_labels = len(label_list) + 1

    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    print(args.config_name)
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    config_ = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config_)

    model = BertECTagging.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, 'train')
        global_step, tr_loss, best_model_dir = train(args, train_dataset, model, tokenizer, label_map)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        if args.do_eval:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            # model = model_class.from_pretrained(best_model_path)
            model = BertECTagging.from_pretrained(best_model_dir, num_labels=num_labels)
            model.to(args.device)
            test_acc, test_precision, test_recall, test_f1 = evaluate(args, model, tokenizer, 'test', label_map)
            logger.info('>> test_acc: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}, test test_f1: {:.4f}'.format(test_acc, test_precision, test_recall, test_f1))

    # Evaluation
    elif args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = BertECTagging.from_pretrained(args.output_dir, num_labels=num_labels)
        model.to(args.device)
        test_acc, test_precision, test_recall, test_f1 = evaluate(args, model, tokenizer, 'dev', label_map)
        logger.info('>> test_acc: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}, test test_f1: {:.4f}'.format(test_acc, test_precision, test_recall, test_f1))

            # checkpoints = [args.output_dir]
            # # if args.eval_all_checkpoints:
            # #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # #     logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            # logger.info("Evaluate the following checkpoints: %s", checkpoints)
            # for checkpoint in checkpoints:
            #     # global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            #     model = model_class.from_pretrained(checkpoint)
            #     model = BertECTagging.from_pretrained(checkpoint, num_labels=num_labels)
            #     model.to(args.device)
            #     # label_map = {i : label for i, label in enumerate(label_list,1)}
            #     result = evaluate(args, model, tokenizer, prefix=global_step, label_map=label_map)
            #     #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #     # results.update(result)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     model.to(args.device)

    #     # Evaluation
    #     results = {}
    #     if args.do_eval and args.local_rank in [-1, 0]:
    #         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #         checkpoints = [args.output_dir]
    #         if args.eval_all_checkpoints:
    #             checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #             logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #         logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #         for checkpoint in checkpoints:
    #             global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #             model = model_class.from_pretrained(checkpoint)
    #             model = BertECTagging.from_pretrained(checkpoint, num_labels=num_labels)
    #             model.to(args.device)
    #             # label_map = {i : label for i, label in enumerate(label_list,1)}
    #             result = evaluate(args, model, tokenizer, prefix=global_step, label_map=label_map)
    #             #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #             # results.update(result)
    
    # # Evaluation
    # results = {}
    # if args.do_eval:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model = BertECTagging.from_pretrained(checkpoint, num_labels=num_labels)
    #         model.to(args.device)
    #         # label_map = {i : label for i, label in enumerate(label_list,1)}
    #         result = evaluate(args, model, tokenizer, prefix=global_step, label_map=label_map)
    #         #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         # results.update(result)

    # return results

if __name__ == "__main__":
    main()
