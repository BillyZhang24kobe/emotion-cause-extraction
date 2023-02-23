from config import *
from transformers import BertForSequenceClassification

import argparse
from utils import *
from dataset import *
from train import *
from predict import Evaluator
from model import BertECTagging, CometBertECTagging, BertEmotion, BertClauseECTagging

from pytorch_transformers import BertConfig, BertForTokenClassification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'comet-bert': (BartConfig, BartForConditionalGeneration, BartTokenizer)
}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.") 
    parser.add_argument("--evaluation_metrics", default='EESE+ECSE', type=str, required=True, help="Whether to evaluate using token-level-cause metric.")
    parser.add_argument("--model_class", default='bert', type=str, required=True, help="Specify the target model.")
    
    ## Other parameters
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--continue_dir", default=None, type=str, help="The model directory where training will continue.")
    parser.add_argument("--log_dir", default=None, type=str, help="The log directory to store model checkpoints' scores for each run.")
    parser.add_argument("--repeat", default=0, type=int, help="The index of model run.")
    parser.add_argument("--device", default=0, type=int, help="The gpu device.")      
    parser.add_argument("--task_name", default="eca", type=str,
                        help="The name of the task to train")
    parser.add_argument("--comet_file", default="xReact", type=str,
                        help="The name of the comet relation.")
    parser.add_argument("--gpt3_shot_type", default="TRS-2", type=str,
                        help="The shot type of gpt3 generated explanations.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="Path to pre-trained BERT model or name")
    parser.add_argument("--comet_model", default=None, type=str, help="Path to pre-trained COMET model or name")
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
    parser.add_argument("--max_comet_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization for comet model. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--print_report", action='store_true', help="Whether to print the classification report")
    parser.add_argument("--print_prediction", action='store_true', help="Whether to print the prediction results")
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

    # args.output_dir = OUTPUT_DIR
    # args.continue_dir = CONTINUE_DIR
    # args.bert_model = BERT_MODEL
    # args.max_seq_length = MAX_SEQ_LENGTH
    # args.max_comet_seq_length = MAX_COMET_LENGTH
    # args.per_gpu_train_batch_size = config.BATCH_SIZE
    # args.per_gpu_eval_batch_size = config.EVAL_BATCH_SIZE
    # args.learning_rate = config.LEARNING_RATE
    # args.comet_file = config.COMET_FILE
    # args.gpt3_shot_type = config.GPT3_SHOT_TYPE

    if args.repeat != 0:
        if os.path.exists(args.log_dir+args.evaluation_metrics+'_'+str(args.seed)+'_'+str(args.repeat - 1)+'.txt'):
            with open(args.log_dir+args.evaluation_metrics+'_'+str(args.seed)+'_'+str(args.repeat - 1)+'.txt', 'r') as f:
                args.continue_dir = f.readlines()[-1].strip()
                logger.info(">>Loading the best model checkpoint from last run %s", args.continue_dir)
        elif os.path.exists(args.log_dir+args.evaluation_metrics+'_'+str(args.seed)+'_'+str(args.repeat - 2)+'.txt'):
            with open(args.log_dir+args.evaluation_metrics+'_'+str(args.seed)+'_'+str(args.repeat - 2)+'.txt', 'r') as f:
                args.continue_dir = f.readlines()[-1].strip()
                logger.info(">>Loading the best model checkpoint from last run %s", args.continue_dir)
        else:
            with open(args.log_dir+args.evaluation_metrics+'_'+str(args.seed)+'_'+str(args.repeat - 3)+'.txt', 'r') as f:
                args.continue_dir = f.readlines()[-1].strip()
                logger.info(">>Loading the best model checkpoint from last run %s", args.continue_dir)

    if 'comet' in args.model_class:
        args.comet_model = COMET_MODEL
        comet_config, comet_model_class, comet_tokenizer_class = MODEL_CLASSES['comet-bert']
        comet_tokenizer = comet_tokenizer_class.from_pretrained(args.comet_model)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')
    #     args.n_gpu = 1
    args.n_gpu = torch.cuda.device_count()
    # args.device = DEVICE
    DEVICE = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, DEVICE, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    processor = ECAProcessor()  # bert processor
    if args.task_name in ['eca', 'eca-clause', 'eca-comet']:
        label_list = processor.get_labels()[0]  # [0] token_labels
        label_map = {i : label for i, label in enumerate(label_list,1)}  # token_labels
        num_labels = len(label_list) + 1
    elif args.task_name == 'emotion_clf':
        label_list = processor.get_labels()[1]  # [1] emotion_labels
        label_map = {i : label for i, label in enumerate(label_list)}  # emotion_labels
        num_labels = len(label_list)

    bert_config_class, bert_model_class, bert_tokenizer_class = MODEL_CLASSES['bert']
    print(args.config_name)
    bert_tokenizer = bert_tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.bert_model, do_lower_case=args.do_lower_case)

    tokenizers = [bert_tokenizer]
    if 'comet' in args.model_class:
        tokenizers.append(comet_tokenizer)

    if args.model_class == 'bert' or args.model_class == 'bert-gpt3':
        model = BertECTagging.from_pretrained(args.bert_model, num_labels=num_labels)
    elif args.model_class == 'bert-clause':
        model = BertClauseECTagging(args, num_labels=num_labels)
        if args.continue_dir:
            model.load_state_dict(torch.load(os.path.join(args.continue_dir, 'model_weights.pth')))
    elif args.model_class == 'comet-bert':
        model = CometBertECTagging(args, comet_config, comet_model_class, comet_tokenizer_class, num_labels)
        if args.continue_dir:
            model.load_state_dict(torch.load(os.path.join(args.continue_dir, 'model_weights.pth')))
    elif args.model_class == 'bert-emotion':
        # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
        model = BertEmotion.from_pretrained(args.bert_model, num_labels=num_labels)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.model_class == 'bert':
            train_dataset = load_and_cache_examples(args, tokenizers, 'train')
        elif args.model_class == 'bert-gpt3':
            train_dataset = load_and_cache_gpt3_examples(args, tokenizers, 'train')
        else:
            train_dataset = ClauseDataset(args, 'train', tokenizers)

        global_step, tr_loss, best_model_dir = train(args, train_dataset, model, tokenizers, label_map)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    elif args.do_eval:
        bert_tokenizer = bert_tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.bert_model, do_lower_case=args.do_lower_case)
        # bert_tokenizer = bert_tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        if args.model_class == 'bert' or args.model_class == 'bert-gpt3':
            model = BertECTagging.from_pretrained(args.output_dir, num_labels=num_labels)
            tokenizers = [bert_tokenizer]
            model.eval()
        elif args.model_class == 'bert-clause':
            model = BertClauseECTagging(args, num_labels=num_labels)
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model_weights.pth')))
            model.eval()
        elif args.model_class == 'comet-bert':
            model = CometBertECTagging(args, comet_config, comet_model_class, comet_tokenizer_class, num_labels)
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model_weights.pth')))
            model.eval()
        elif 'emotion' in args.model_class:
            tokenizers = [bert_tokenizer]
            model = BertEmotion(args, num_labels)
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model_weights.pth')))
            model.eval()
            
        model.to(args.device)
        evaluator = Evaluator(args, model, tokenizers, 'dev', label_map)
        test_acc, test_precision, test_recall, test_f1 = evaluator.evaluate(args)
        logger.info('>> test_acc: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_precision, test_recall, test_f1))

if __name__ == "__main__":
    main()
