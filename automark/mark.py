import numpy as np
import torch
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer

class MarkingHead(torch.nn.Module):
    def __init__(self, dimension):
        self.fc1 = torch.nn.Linear(dimension, dimension)
        self.prediction = torch.nn.Linear(dimension, 2)
    
    def forward(self, embedding):
        x = F.relu(self.fc1(embedding))
        x = F.log_softmax(self.prediction(x), dim=-1)
        return x

class AutoMark(torch.nn.Module):
    def __init__(self, model='bert-base-multilingual-cased', cuda=False):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.bert = BertModel.from_pretrained(model)
        self.marking_head = MarkingHead(self.bert.hidden_size)

    def forward(self, sentence, mask):
        embeddings = self.bert(sentence, token_type_ids=mask)
        #embeddings should be (Batch, Len, Emb_Dim)
        shape = embeddings.shape
        embeddings = embeddings.view(embeddings.shape[0], -1)
        flat_markings = self.marking_head(embeddings)
        markings = flat_markings.view(shape)
        return markings