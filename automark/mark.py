import numpy as np
import torch
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer


class MarkingHead(torch.nn.Module):
    def __init__(self, dimension, hidden_dimension, activation, bias):
        super(MarkingHead, self).__init__()

        self.fc1 = torch.nn.Linear(dimension, hidden_dimension)
        # Output 0 = not marked, Output 1 = marked, Output 2 = ignore
        self.prediction = torch.nn.Linear(hidden_dimension, 2, bias=bias)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "selu":
            self.activation = F.selu

    def forward(self, embedding):
        x = self.activation(self.fc1(embedding))
        # print("Avg activation {:.2f}".format(torch.mean(x)))
        # print("Min activation {:.2f}".format(torch.min(x)))
        # print("Max activation {:.2f}".format(torch.max(x)))
        x = F.log_softmax(self.prediction(x), dim=-1)
        # predictions = torch.argmax(x, dim=-1)
        # print("1s: {:.2f}%".format(
        #    predictions.sum().detach()/predictions.numel()*100))
        return x


class AutoMark(torch.nn.Module):
    def __init__(
        self,
        model="bert-base-multilingual-cased",
        freeze_bert=False,
        hidden_dimension=768,
        activation="relu",
        bias=False,
        cuda=False,
    ):
        super(AutoMark, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.bert = BertModel.from_pretrained(model)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.marking_head = MarkingHead(
            self.bert_hidden_size, hidden_dimension, activation=activation, bias=bias
        )
        self.pad_index = self.tokenizer.vocab.get("[PAD]")
        self.unk_index = self.tokenizer.vocab.get("[UNK]")
        self.bos_index = self.tokenizer.vocab.get("[CLS]")
        self.sep_index = self.tokenizer.vocab.get("[SEP]")
        if freeze_bert:
            self.freeze_bert()

    def forward(self, sentence, label_mask, attention_mask):
        src_trg, lens = sentence
        embeddings = self.bert(
            src_trg, token_type_ids=label_mask, attention_mask=attention_mask
        )[0]
        # embeddings should be (Batch, Len, Emb_Dim)
        shape = embeddings.shape
        # print("shape embeddings {}".format(embeddings.shape))
        embeddings = embeddings.view(-1, self.bert_hidden_size)
        flat_markings = self.marking_head(embeddings)
        markings = flat_markings.view(shape[0], shape[1], -1)
        # print("shape markings {}".format(markings.shape))
        return markings

    def get_loss_for_batch(self, batch, loss_function, weighting):
        assert batch.src_trg[0].shape == batch.id_mask.shape
        predictions = self.forward(batch.src_trg, batch.id_mask, batch.attention_mask)
        labels = batch.weights
        mask = batch.label_mask

        if weighting == "percentage":
            print("percentage")
            weights = torch.ones_like(batch.loss_weight)
            total_ones = torch.sum(labels, dim=1).float()
            total_trgs = batch.trg_len.float()

            one_weights = 1 - (total_ones / total_trgs)
            zero_weights = 1 - one_weights

            ones_weight_matrix = (
                batch.id_mask.float() * labels.float() * one_weights.view(-1, 1)
            )

            zero_weight_matrix = (
                batch.id_mask.float()
                * ((-1 * labels.float()) + 1)
                * zero_weights.view(-1, 1)
            )

            weight_matrix = ones_weight_matrix + zero_weight_matrix
            weights = weight_matrix + 1e-6
        elif weighting == "constant":
            print("constant")
            weights = batch.loss_weight
        elif weighting == "wrong":
            argmax_predictions = predictions.argmax(dim=-1)

            wrong_predictions = argmax_predictions != batch.weights

            weights = torch.ones_like(batch.loss_weight)

            weights = torch.where(wrong_predictions, weights * 2, weights * 0.1)
        else:
            print("none")
            weights = torch.ones_like(batch.loss_weight)

        loss = loss_function(predictions, labels, mask, weights)

        loss = loss.view(batch.src_trg[0].shape[0], -1)

        batch_loss = torch.sum(loss)

        # proportion of ones in the prdictions
        num_valid_tokens = batch.label_mask.sum().detach()
        pred_labels = predictions.argmax(-1)
        num_one_pred = (pred_labels.float() * batch.label_mask).sum().detach()
        ones = num_one_pred / num_valid_tokens
        # accuracy: ratio of correct predictions
        num_corr_pred = (
            ((pred_labels == labels).float() * batch.label_mask).detach().sum()
        )
        acc = num_corr_pred / num_valid_tokens
        return batch_loss, ones, acc, predictions

    def freeze_bert(self):
        for name, param in self.named_parameters():
            if "bert" in name:
                param.requires_grad = False


def build_model(config):
    model = AutoMark(
        config["bert"]["path"],
        cuda=config["train"]["cuda"],
        hidden_dimension=config["model"]["hidden_dimension"],
        activation=config["model"]["activation"],
        freeze_bert=config["model"]["freeze_bert"],
        bias=config["model"]["head_bias"],
    )
    return model
