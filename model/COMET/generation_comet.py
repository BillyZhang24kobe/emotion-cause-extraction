import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, BartConfig, BartTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch

from IPython import embed
import csv

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.config = BartConfig.from_pretrained(model_path, output_hidden_states=True)
        self.model = BartForConditionalGeneration.from_pretrained(model_path, config=self.config).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )
                
                # print(summaries)
                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


if __name__ == "__main__":

    # sample usage
    print("model loading ...")
    comet = Comet("./comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")

    # load data
    # folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
    # for fold in folds:
    #     print(fold)
    for data_partition in ['train', 'dev', 'test']:
        file_name = '../../data/goodnewseveryone-v1.0/data/{}.tsv'.format(data_partition)
        data_file = open(file_name)
        data_tsv = csv.reader(data_file, delimiter="\t")

        rel = 'xEffect'
        writeToFile = './data/goodnewseveryone/eca-{}-cleaned-{}.tsv'.format(data_partition, rel)
        rel_col_name = rel

        with open(writeToFile, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['document', 'token_label', 'emotion-label', rel_col_name])
            for i, line in enumerate(data_tsv):
                if i == 0: continue
                document = line[0]
                token_label = line[1]
                emotion_label = line[2]

                queries = []
                head = document.strip()
                x_response = ''
                query = "{} {} [GEN]".format(head, rel)
                queries.append(query)
                results = comet.generate(queries, decode_method="beam", num_generate=5)
                x_response += results[0][0].strip()
                tsv_writer.writerow([document, token_label, emotion_label, x_response])