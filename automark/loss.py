import torch
from torch import nn
from torch.nn import functional as F 

class BXentLoss(nn.Module):
    def __init__(self, ignore_id=-1):
        super(BXentLoss, self).__init__()
        self.criterion = torch.nn.BCELoss(reduce=None, reduction='none')

    def forward(self, predictions, labels, mask, weights=None):
        flat_predictions = predictions.view(-1)
        flat_labels = labels.view(-1).float()
        flat_mask = mask.view(-1)

        loss = self.criterion(flat_predictions, flat_labels)
        masked_loss = loss * flat_mask
        if weights is not None:
            masked_loss = masked_loss * weights.view(-1)
        return masked_loss

class XentLoss(nn.Module):
    def __init__(self, ignore_id=-1):
        super(XentLoss, self).__init__()
        self.ignore_id = ignore_id
        self.criterion = torch.nn.NLLLoss(ignore_index=self.ignore_id,
                                          reduction='none')

    def forward(self, predictions, labels, mask, weights=None):
        flat_predictions = predictions.view(-1, predictions.shape[-1])
        flat_labels = labels.view(-1)
        flat_mask = mask.view(-1)

        loss = self.criterion(flat_predictions, flat_labels)
        masked_loss = loss * flat_mask

        if weights is not None:
            masked_loss *= weights.view(-1)
        return masked_loss
