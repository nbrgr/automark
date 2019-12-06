import numpy as np
import torch
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer

class MarkingHead(torch.nn.Module):
    def __init__(self, dimension):
        super(MarkingHead, self).__init__()

        self.fc1 = torch.nn.Linear(dimension, dimension)
        #Output 0 = not marked, Output 1 = marked, Output 2 = ignore
        self.prediction = torch.nn.Linear(dimension, 3)
    
    def forward(self, embedding):
        x = F.relu(self.fc1(embedding))
        x = F.log_softmax(self.prediction(x), dim=-1)
        return x

class AutoMark(torch.nn.Module):
    def __init__(self, model='bert-base-multilingual-cased', cuda=False):
        super(AutoMark, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.bert = BertModel.from_pretrained(model)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.marking_head = MarkingHead(self.bert_hidden_size)
        self.pad_index = self.tokenizer.vocab.get('[PAD]')
        self.unk_index = self.tokenizer.vocab.get('[UNK]')
        self.bos_index = self.tokenizer.vocab.get('[CLS]')
        self.sep_index = self.tokenizer.vocab.get('[SEP]')

    def forward(self, sentence, mask):
        
        src_trg, lens = sentence
        mask = torch.tensor(mask)
        print(src_trg, mask)
        embeddings = self.bert(src_trg, token_type_ids=mask)[0]
        #embeddings should be (Batch, Len, Emb_Dim)
        shape = embeddings.shape
        embeddings = embeddings.view(-1, self.bert_hidden_size)
        flat_markings = self.marking_head(embeddings)
        markings = flat_markings.view(shape[0], shape[1], -1)
        print(markings.shape)
        return markings

    def get_loss_for_batch(self, batch, loss_function):
        predictions = self.forward(batch.src_trg, batch.attention_mask)
        labels = batch.weights
        mask = batch.label_mask
        batch_loss = torch.sum(loss_function(predictions, labels, mask))
        return batch_loss

def build_model(config):
    return AutoMark(config['bert']['path'], config['train']['cuda'])