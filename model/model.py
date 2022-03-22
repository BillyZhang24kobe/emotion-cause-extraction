from turtle import forward
from pytorch_transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel
from transformers import BartConfig, PreTrainedModel, PretrainedConfig

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class BertECTagging(BertForTokenClassification):
    def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None):

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)

        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class CometBertECTagging(nn.Module):
    def __init__(self, args, comet_config, comet_model_class, comet_tokenizer_class):
        super(CometBertECTagging, self).__init__()
        # self.device = args.device
        self.bert_config = BertConfig.from_pretrained(args.bert_model)
        self.num_labels = self.bert_config.num_labels
        self.bert_model = BertModel(self.bert_config)
        classifier_dropout = (self.bert_config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.bert_config.num_labels)
        self.comet_config = comet_config.from_pretrained(args.comet_model, output_hidden_states=True)
        self.comet_model = comet_model_class.from_pretrained(args.comet_model, config=self.comet_config)
        self.comet_tokenizer = comet_tokenizer_class.from_pretrained(args.comet_model)
    
    def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None):
        bert_input_ids = input_ids[:,:args.max_seq_length]
        comet_input_ids = input_ids[:,args.max_seq_length:]
        # get BERT encodings
        bert_sequence_output, _ = self.bert_model(bert_input_ids, 
                                                  token_type_ids, 
                                                  attention_mask)

        batch_size,max_len,feat_dim = bert_sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = bert_sequence_output[i][j]
        
        bert_sequence_output = self.dropout(valid_output)  # bert embeddings: batch_size x max_len x d_model(768)

        # get comet encodings
        _, comet_all_layers_outputs = self.comet_model(comet_input_ids)

        encoder_last_hidden_states = comet_all_layers_outputs[-1]  # batch_size x max_len x d_model(1024)
        comet_sequence_output = encoder_last_hidden_states

        # combine BERT and COMET
        combined_output = torch.cat((bert_sequence_output, comet_sequence_output), dim=2)  # batch_size x max_len x (bert_dim+comet_dim)

        logits = self.classifier(combined_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
