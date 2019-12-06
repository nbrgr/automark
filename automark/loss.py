import torch
from torch import nn
from torch.nn import functional as F 

class XentLoss(nn.Module):
    def __init__(self, ignore_id):
        super(XentLoss, self).__init__()
        self.ignore_id = ignore_id
        self.criterion = torch.nn.NLLLoss(ignore_index=self.ignore_id, reduction='sum')

    def forward(self, predictions, labels, mask):
        predictions = predictions.view(predictions.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)
        mask = mask.view(labels.shape[0], -1)

        masked_predictions = predictions * mask

        loss = self.criterion(masked_predictions, labels)

        return loss