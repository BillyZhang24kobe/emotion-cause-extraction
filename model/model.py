from turtle import forward
# from config import DEVICE
from pytorch_transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel
from transformers import BartConfig, PreTrainedModel, PretrainedConfig, BertForSequenceClassification

import torch
from torch import device, nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import numpy as np
from typing import Optional, Tuple

from IPython import embed

class CometAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim  # bert encoder output
        self.decoder_dim = decoder_dim  # comet last hidden state vector

    def forward(self,
        query: torch.Tensor,  # [batch_size, decoder_dim] -> comet_dim -> 1024
        values: torch.Tensor,  # [batch_size, seq_length, encoder_dim] -> seq_length x 768
        ):
        # embed()
        output = []
        weights = self._get_weights(query, values)  # [batch_size, seq_length]
        weights = nn.functional.softmax(weights, dim=0)
        output_ = torch.zeros((weights.shape[1], values.shape[2])).to(DEVICE)
        for batch in range(query.shape[0]):
            weights_b = weights[batch]
            for i in range(weights_b.shape[0]):
                output_[i] = values[batch][i] * weights_b[i]
            output.append(output_)
        
        return torch.stack(output).to(DEVICE)

class AdditiveAttention(CometAttention):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1)).to(DEVICE)
        self.W_1 = nn.Linear(self.decoder_dim, self.decoder_dim).to(DEVICE)
        self.W_2 = nn.Linear(self.encoder_dim, self.decoder_dim).to(DEVICE)

    def _get_weights(self,
        query: torch.Tensor,  # [decoder_dim] = 1024
        values: torch.Tensor,  # [seq_length, encoder_dim]
    ):
        query = query.unsqueeze(1).repeat(1, values.size(1), 1)  # [batch_size, seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [batch_size, seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]

class MultiplicativeAttention(CometAttention):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.W = torch.nn.Parameter(torch.FloatTensor(
            self.decoder_dim, self.encoder_dim).uniform_(-0.1, 0.1)).to(DEVICE)

    def _get_weights(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
    ):
        # embed()
        result = query @ self.W
        result = result.unsqueeze(1)
        result = result @ torch.transpose(values, 1, 2)  
        weights = result.squeeze(1)  # [batch_size, seq_length]
        return weights/np.sqrt(self.decoder_dim)  # [seq_length]


class BAttention(nn.Module):
    """
    Compute (bahdanau) attention mechanism on the output features.

    Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
    """
    def __init__(self, hidden_dim: int) -> None:
        super(BAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1)).to(DEVICE)
        self.score_proj = nn.Linear(hidden_dim, 1).to(DEVICE)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn


class SelfAttention(nn.Module):
    """
    Self-attention model
    """
    def __init__(self, emb_dim, num_heads=8, hidden_size=128):
        super(SelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # self.embeddings = embeddings  # [batch_size, # of clauses, 768]
        self.k = nn.Linear(self.emb_dim, self.hidden_size * self.num_heads, bias=False).to(DEVICE)  # key matrix
        self.v = nn.Linear(self.emb_dim, self.hidden_size * self.num_heads, bias=False).to(DEVICE)  # value matrix
        self.q = nn.Linear(self.emb_dim, self.hidden_size * self.num_heads, bias=False).to(DEVICE)  # query matrix

        self.final = nn.Linear(self.hidden_size * self.num_heads, self.emb_dim).to(DEVICE)  # final linear layer

    # x is a sequence of clauses in a batch
    def forward(self, x):

        b, t, _ = x.shape
        e = self.hidden_size
        h = self.num_heads
        keys = self.k(x).view(b, t, h, e)
        values = self.v(x).view(b, t, h, e)
        queries = self.q(x).view(b, t, h, e)

        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        queries = queries.transpose(2, 1)

        dot = queries @ keys.transpose(3, 2)
        dot = dot / np.sqrt(e)
        dot = F.softmax(dot, dim=3)

        out = dot @ values
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.final(out)

        return out

        # d_k = self.hidden_size
        # k = self.k(x)  # batch_size * seq_len * hidden_size
        # v = self.v(x)  # batch_size * seq_len * hidden_size
        # q = self.q(x)  # batch_size * seq_len * hidden_size

        # output = torch.bmm(q, torch.transpose(k, 1, 2)) / np.sqrt(d_k)  # Q * K^T = batch_size * seq_len * seq_len
        # output = F.softmax(output, dim=2)  # batch_size * seq_len * seq_len
        # output = torch.bmm(output, v)  # batch_size * seq_len * hidden_size

        # output = torch.relu(output)

        # return output

        # # pooling layer
        # output = torch.max(output, dim=1)[0]  # shape: batch_size * hidden_size

        # output = self.final(output)  # batch_size * num_classes

        # output = torch.relu(output)

        # return output


def CometID_indexing(comet_input_ids, device):
    output = []
    for c_input in comet_input_ids:
        rel_idx = (c_input == 50313).nonzero(as_tuple=True)[0]  # index of xReact relation in the current tokenized input
        output.append(rel_idx - 1)

    return torch.stack(output).to(device)

def BertID_indexing(clause_bert_ids, doc_bert_ids, device):
    output = []
    # debug = []
    for clause, doc in zip(clause_bert_ids, doc_bert_ids):
        # print(clause)
        flag = False
        non_ones_idx = torch.where(clause != 1)[0]  # ignore [unused0] token
        clause = clause[non_ones_idx]
        len_c = torch.nonzero(clause, as_tuple=True)[0].shape[0]
        cand_ids = (i for i, d_id in enumerate(doc) if d_id == clause[0])
        for ind in cand_ids:
            cl_nz_indices = torch.nonzero(clause, as_tuple=True)[0]  # clause nonzero indices
            if doc[ind:ind+len_c].tolist() == torch.index_select(clause, 0, cl_nz_indices).tolist():
                output.append(torch.tensor([ind, ind+len_c]))
                flag = True
                break

        if not flag:
            output.append(torch.tensor([0, clause_bert_ids.shape[1]]))
        #     embed()

    return torch.stack(output).to(device)

def BertEmbedding_indexing(args, doc_bert_embedding, clause_indexing):
    output = []
    clause_lengths = []
    for doc_bert, idx in zip(doc_bert_embedding, clause_indexing):
        start = idx.tolist()[0]
        end = idx.tolist()[1]
        clause_len = end - start
        clause_lengths.append(torch.tensor([clause_len]).to(args.device))
        clause_embedding = doc_bert[start:end, :]
        padding = torch.zeros(args.max_comet_seq_length-clause_len, doc_bert.shape[1]).to(args.device)  # TODO: might be a problem???
        output.append(torch.cat((clause_embedding, padding), dim=0))
        # pad clause seq_length to max_comet_seq_length

    return torch.stack(output).to(args.device), torch.stack(clause_lengths).to(args.device)


def CometEmbedding_indexing(args, comet_encoder_last_hidden_state, indexing_last_comet_token):
    output = []
    for comet_last_state, last_index in zip(comet_encoder_last_hidden_state, indexing_last_comet_token):
        output.append(comet_last_state[last_index, :].squeeze(0))
    
    return torch.stack(output).to(args.device)


def CometEmbedding_repeat_and_padding(args, comet_encoder_last_hidden_state, clause_lengths):
    output = []

    # TODO: comet_clause_length
    comet_encoder_last_token_hidden_state = comet_encoder_last_hidden_state[:, -1, :].unsqueeze(1)  # TODO: take the meaningful token?
    
    for clause_len, comet_encoder_last in zip(clause_lengths, comet_encoder_last_token_hidden_state):
        comet_repeat = comet_encoder_last.repeat(clause_len, 1)
        padding = torch.zeros(args.max_comet_seq_length-clause_len, comet_encoder_last.shape[1]).to(args.device)
        output.append(torch.cat((comet_repeat, padding), dim=0))

    return torch.stack(output).to(args.device) 
    # comet_sequence_output = comet_sequence_output[:, -1, :].unsqueeze(1).repeat(1, args.max_comet_seq_length, 1)



class BertECTagging(BertForTokenClassification):
    def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)

        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32, device=device)  # TODO: maybe do not need valid_output?
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # embed()
            # ['[PAD]', "O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
            # class_weights = [0, 0.006523830013414665, 0.312046004842615, 0.04602473103879291, 
            #                  0.3381436536569367, 1.0, 0, 0]  # in training data
            class_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0]  # in training data
            loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            # loss_fct = CrossEntropyLoss(ignore_index=0)
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


class BertClauseECTagging(nn.Module):
    def __init__(self, args, num_labels):
        super(BertClauseECTagging, self).__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_model)
        self.num_labels = num_labels
        self.bert_model = BertModel(self.bert_config)
        classifier_dropout = (self.bert_config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        self.lstm = nn.LSTM(self.bert_config.hidden_size, self.bert_config.hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.classifier = nn.Linear(self.bert_config.hidden_size * 3, self.num_labels)
        # self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        # self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.num_labels)
    
    def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None):
        clause_input_ids = input_ids[:, :args.max_seq_length]  # batch_size x max_seq_length
        # comet_input_ids = input_ids[:, args.max_seq_length:]  # batch_size x max_comet_length

        # get BERT encodings - clause bert embeddings [batch_size, max_seq_len, 768]
        clause_sequence_output, _ = self.bert_model(clause_input_ids, 
                                                  token_type_ids, 
                                                  attention_mask)

        # clause_sequence_output = clause_sequence_output.view(-1, 768).unsqueeze(0)

        # taking [CLS] as the pooler output for each clause
        # pooled_output = clause_sequence_output[:,0,:].unsqueeze(0)  # [1, # of clause, 768]

        ###### attention-based pooling ######
        attention = BAttention(self.bert_config.hidden_size)
        pooled_output, _ = attention(clause_sequence_output, clause_sequence_output, clause_sequence_output)
        pooled_output = pooled_output.squeeze(1).unsqueeze(0)

        ###### bidirectional lstm to capture clause-level interactions via concatenation ######
        lstm_output, _ = self.lstm(pooled_output)  # [1, # of clause, 768 x 2]
        lstm_output = lstm_output.squeeze(0).unsqueeze(1).repeat(1, clause_sequence_output.size(1), 1)  # [# of clause, max_len, 768 x 2]
        clause_sequence_output = torch.cat((clause_sequence_output, lstm_output), 2)  # [# of clauses, max_len, 768 x 2 + 768]

        # Self-attention layer to capture clausal interactions
        # self_attention = SelfAttention(self.bert_config.hidden_size)
        # self_attn_output = self_attention(pooled_output)  # [batch_size, # of clauses, hidden_size]
        # # clause_sequence_output = self_attn_output.view(-1, args.max_seq_length, self.bert_config.hidden_size)
        # self_attn_output = self_attn_output.squeeze(0).unsqueeze(1).repeat(1, clause_sequence_output.size(1), 1)
        # clause_sequence_output = torch.cat((clause_sequence_output, self_attn_output), 2)
        # attention = AdditiveAttention(encoder_dim=self.bert_config.hidden_size, decoder_dim=self_attn_output.shape[-1])  # Attention calculation for each clause
        # clause_sequence_output = attention(self_attn_output, clause_sequence_output)  # [batch_size, max_seq_length, bert_dim]

        combined_output = self.dropout(clause_sequence_output)
        # combined_output = self.dropout(combined_output)
        # combined_output = clause_sequence_output
        logits = self.classifier(combined_output)

        if labels is not None:
            # class_weights = [0, 6.20332001687303e-06, 0.0002981514609421586, 4.396377385034731e-05, 
            #                  0.00032299741602067185, 0.0009541984732824427, 0, 0]  # in training data
            # loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            # class_weights = [0, 0.016, 0.33, 0.045, 0.333, 0.333, 0, 0]

            class_weights = [0, 1, 1, 1, 1, 1, 0, 0]  # ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
            loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            
            # loss_fct = CrossEntropyLoss(ignore_index=0)
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
    def __init__(self, args, comet_config, comet_model_class, comet_tokenizer_class, num_labels):
        super(CometBertECTagging, self).__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_model)
        self.num_labels = num_labels
        self.bert_model = BertModel(self.bert_config)
        classifier_dropout = (self.bert_config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.comet_config = comet_config.from_pretrained(args.comet_model, output_hidden_states=True)
        self.comet_model = comet_model_class.from_pretrained(args.comet_model, config=self.comet_config)
        self.comet_tokenizer = comet_tokenizer_class.from_pretrained(args.comet_model)
        self.comet_to_bert = nn.Linear(self.comet_config.d_model, self.bert_config.hidden_size)

        self.lstm = nn.LSTM(self.bert_config.hidden_size, self.bert_config.hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.classifier = nn.Linear(self.bert_config.hidden_size+self.comet_config.d_model, self.num_labels)
        self.classifier = nn.Linear(self.bert_config.hidden_size * 3, self.num_labels)
        # self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        # self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.num_labels)
    
    def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None):
        clause_input_ids = input_ids[:, :args.max_seq_length]  # batch_size x max_seq_length
        # comet_input_ids = input_ids[:, args.max_seq_length:]  # batch_size x max_comet_length

        # get BERT encodings - clause bert embeddings [batch_size, max_seq_len, 768]
        clause_sequence_output, _ = self.bert_model(clause_input_ids, 
                                                  token_type_ids, 
                                                  attention_mask)

        # clause_sequence_output = clause_sequence_output.view(-1, 768).unsqueeze(0)

        # taking [CLS] as the pooler output for each clause
        # pooled_output = clause_sequence_output[:,0,:].unsqueeze(0)  # [1, # of clause, 768]

        ###### attention-based pooling ######
        attention = BAttention(self.bert_config.hidden_size)
        pooled_output, _ = attention(clause_sequence_output, clause_sequence_output, clause_sequence_output)
        pooled_output = pooled_output.squeeze(1).unsqueeze(0)

        ###### bidirectional lstm to capture clause-level interactions via concatenation ######
        lstm_output, _ = self.lstm(pooled_output)  # [1, # of clause, 768 x 2]
        lstm_output = lstm_output.squeeze(0).unsqueeze(1).repeat(1, clause_sequence_output.size(1), 1)  # [# of clause, max_len, 768 x 2]
        clause_sequence_output = torch.cat((clause_sequence_output, lstm_output), 2)  # [# of clauses, max_len, 768 x 2 + 768]

        # Self-attention layer to capture clausal interactions
        # self_attention = SelfAttention(self.bert_config.hidden_size)
        # self_attn_output = self_attention(pooled_output)  # [batch_size, # of clauses, hidden_size]
        # # clause_sequence_output = self_attn_output.view(-1, args.max_seq_length, self.bert_config.hidden_size)
        # self_attn_output = self_attn_output.squeeze(0).unsqueeze(1).repeat(1, clause_sequence_output.size(1), 1)
        # clause_sequence_output = torch.cat((clause_sequence_output, self_attn_output), 2)
        # attention = AdditiveAttention(encoder_dim=self.bert_config.hidden_size, decoder_dim=self_attn_output.shape[-1])  # Attention calculation for each clause
        # clause_sequence_output = attention(self_attn_output, clause_sequence_output)  # [batch_size, max_seq_length, bert_dim]

        # indexing the first token before padding in comet_input_ids (i.e. taking the last hidden state from the last meaningful comet token)
        # indexing_last_comet_token = CometID_indexing(comet_input_ids, device)  # [batch_size, 1]

        # freeze comet model parameters
        # for par in self.comet_model.parameters():
        #     par.requires_grad = False
        # comet_outputs = self.comet_model(comet_input_ids)
        # comet_encoder_last_hidden_state = comet_outputs.encoder_last_hidden_state    # batch_size x max_len x d_model(1024)

        ##### Downsize comet embedding to bert size #####
        # comet_sequence_output = CometEmbedding_indexing(args, comet_encoder_last_hidden_state, indexing_last_comet_token) # [batch_size, 1024]
        # comet_encoder_last_hidden_state = self.comet_to_bert(comet_sequence_output.unsqueeze(1))  # batch_size x 1 x 768
        # combined_output = torch.cat((clause_sequence_output, comet_sequence_output), dim=2)  # batch_size x max_len x (bert_dim+comet_dim)

        ##### concatenate BERT and COMET #####
        # comet_sequence_output = CometEmbedding_repeat_and_padding(args, comet_encoder_last_hidden_state, clause_lengths)
        # combined_output = torch.cat((clause_sequence_output, comet_sequence_output), dim=2)  # batch_size x max_len x (bert_dim+comet_dim)
        # combined_output = nn.functional.normalize(combined_output, dim=2)
        
        ##### Attention-based COMET-BERT #####
        # query: [decoder_dim] -> comet_dim -> 1024
        # values: [seq_length, encoder_dim] -> seq_length x 768
        # attention = MultiplicativeAttention(encoder_dim=self.bert_config.hidden_size*2, decoder_dim=self.comet_config.d_model)
        # comet_sequence_output = CometEmbedding_indexing(args, comet_encoder_last_hidden_state, indexing_last_comet_token) # [batch_size, 1024] 
        # combined_output = attention(comet_sequence_output, clause_sequence_output)  # [batch_size, max_seq_length, bert_dim]

        combined_output = self.dropout(clause_sequence_output)
        # combined_output = self.dropout(combined_output)
        # combined_output = clause_sequence_output
        logits = self.classifier(combined_output)

        if labels is not None:
            # class_weights = [0, 6.20332001687303e-06, 0.0002981514609421586, 4.396377385034731e-05, 
            #                  0.00032299741602067185, 0.0009541984732824427, 0, 0]  # in training data
            # loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            # class_weights = [0, 0.016, 0.33, 0.045, 0.333, 0.333, 0, 0]

            class_weights = [0, 1, 1, 1, 1, 1, 0, 0]  # ["O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
            loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            
            # loss_fct = CrossEntropyLoss(ignore_index=0)
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


# self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None
class BertEmotion(BertForSequenceClassification):
    def forward(self, args, device, input_ids, segment_ids, input_mask, labels=None, valid_ids=None, l_mask=None):

        # get BERT encodings - doc bert embeddings [batch_size, max_seq_len, 768]
        bert_output = self.bert(input_ids, segment_ids, input_mask)

        ###### [CLS] pooling ######
        # pooled_output = bert_output.pooler_output  # [CLS] embeddings: [batch_size, 768]

        ###### mean pooling ######
        # bert_output = bert_output.last_hidden_state
        # pooled_output = torch.mean(bert_output, dim=1)  # [batch_size, 768]

        ###### max pooling ######
        # bert_output = bert_output.last_hidden_state
        # pooled_output = torch.max(bert_output, dim=1)[0]  # [batch_size, 768]

        ###### attention-based pooling ######
        bert_output = bert_output.last_hidden_state
        attention = BAttention(self.config.hidden_size)
        pooled_output, _ = attention(bert_output, bert_output, bert_output)
        pooled_output = pooled_output.squeeze(1)

        pooled_output = self.dropout(pooled_output)  # [batch_size, max_seq_len, 768]
        logits = self.classifier(pooled_output)

        if labels is not None:
            # class_weights = [0.0014, 0.0023, 0.0046, 0.0015, 0.0036, 0.0014]  # in training data
            # loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            return loss
        else:
            return logits


# class BertEmotion(nn.Module):
#     def __init__(self, args, num_labels):
#         super(BertEmotion, self).__init__()
#         self.bert_config = BertConfig.from_pretrained(args.bert_model)
#         self.num_labels = num_labels
#         self.bert_model = BertModel(self.bert_config)
#         classifier_dropout = (self.bert_config.hidden_dropout_prob)
#         self.dropout = nn.Dropout(classifier_dropout)

#         self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
    
#     def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):

#         # get BERT encodings - clause bert embeddings [batch_size, max_seq_len, 768]
#         clause_sequence_output, _ = self.bert_model(input_ids, token_type_ids, attention_mask)

#         ###### [CLS] pooling ######
#         pooled_output = clause_sequence_output[:, 0, :]  # [CLS] embeddings: [batch_size, 768]

#         ###### mean pooling ######
#         # pooled_output = torch.mean(clause_sequence_output, dim=1)  # [batch_size, 768]

#         ###### max pooling ######
#         # pooled_output = torch.max(clause_sequence_output, dim=1)[0]  # [batch_size, 768]

#         ###### attention-based pooling ######
#         # attention = BAttention(self.bert_config.hidden_size)
#         # pooled_output, _ = attention(clause_sequence_output, clause_sequence_output, clause_sequence_output)
#         # pooled_output = pooled_output.squeeze(1)

#         pooled_output = self.dropout(pooled_output)  # [batch_size, max_seq_len, 768]
#         logits = self.classifier(pooled_output)

#         if labels is not None:
#             # class_weights = [0.0014, 0.0023, 0.0046, 0.0015, 0.0036, 0.0014]  # in training data
#             # loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits, labels.view(-1))
#             print(labels, loss)
#             return loss
#         else:
#             return logits


# def forward(self, args, device, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):

#         sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)

#         batch_size,max_len,feat_dim = sequence_output.shape
#         valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32, device=device)  # TODO: maybe do not need valid_output?
#         for i in range(batch_size):
#             jj = -1
#             for j in range(max_len):
#                     if valid_ids[i][j].item() == 1:
#                         jj += 1
#                         valid_output[i][jj] = sequence_output[i][j]
        
#         sequence_output = self.dropout(valid_output)
#         logits = self.classifier(sequence_output)

#         if labels is not None:
#             # embed()
#             # ['[PAD]', "O", "B-CAU", "I-CAU",  "B-EMO", "I-EMO", '[CLS]', '[SEP]']
#             class_weights = [0, 0.006523830013414665, 0.312046004842615, 0.04602473103879291, 
#                              0.3381436536569367, 1.0, 0, 0]  # in training data
#             # class_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0]  # in training data
#             loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
#             # loss_fct = CrossEntropyLoss(ignore_index=0)
#             # Only keep active parts of the loss
#             attention_mask_label = None
#             if attention_mask_label is not None:
#                 active_loss = attention_mask_label.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)[active_loss]
#                 active_labels = labels.view(-1)[active_loss]
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# bert_input_ids = torch.tensor([[1,4,5,6,9,10,11,0,0,0,0],[2,5,6,8,1,8,0,0,0,0,0]])
# clause_bert_ids = torch.tensor([[4,5,6,0,0],[2,5,0,0,0]])
# print(BertID_indexing(clause_bert_ids, bert_input_ids).shape)