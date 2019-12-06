import torch
from torch import nn
from torch.nn import functional as F 

class XentLoss(nn.Module):
    def __init__(self, ignore_id):
        super(XentLoss, self).__init__()
        self.ignore_id = ignore_id
        self.criterion = torch.nn.NLLLoss(ignore_index=self.ignore_id, reduction='none')

    def forward(self, predictions, labels, mask):
        flat_predictions = predictions.view(-1, predictions.shape[-1])
        print(flat_predictions.shape)
        flat_labels = labels.view(-1)
        print("Flat labels shape",flat_labels.shape)
        flat_mask = mask.view(-1)
        print(flat_mask.shape)

        loss = self.criterion(flat_predictions, flat_labels)
        print(loss)
        masked_loss = loss * flat_mask

        return masked_loss