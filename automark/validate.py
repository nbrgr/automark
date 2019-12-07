import torch
from automark.dataset import MergeDataset, make_data_iter
import numpy as np 

from sklearn.metrics import f1_score 

def validate_on_data(
    batch_size=1,
    data=None,
    eval_metric=None,
    model=None,
    use_cuda=False,
    loss_function=None,
):

    valid_iter = make_data_iter(data, batch_size, False, False)

    model.eval()

    pred_label_list = []
    label_list = []

    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        total_seqs = 0
        valid_ones = 0
        valid_acc = 0
        for valid_batch in valid_iter:
            batch_loss, ones, acc, predictions = model.get_loss_for_batch(
                valid_batch, loss_function, None
            )
            
            pred_labels = predictions.argmax(-1).view(-1).numpy()
            labels = valid_batch.weights.view(-1).numpy()
            label_mask = valid_batch.id_mask.view(-1).numpy()
            pred_labels_masked = pred_labels[label_mask == 1]
            labels_masked = labels[label_mask == 1]
            pred_label_list.extend(pred_labels_masked.tolist())
            label_list.extend(labels_masked.tolist())
            total_loss += batch_loss
            total_tokens += valid_batch.trg_len.sum().item()
            total_seqs += valid_batch.src_trg[0].shape[0]
            valid_ones += ones
            valid_acc += acc

        valid_ones = valid_ones / total_seqs
        valid_acc = valid_acc / total_seqs
        total_loss = total_loss / total_tokens
    f1 = f1_score(label_list, pred_label_list)

    # Return valid_score, valid_loss, valid_sources, valid_references, valid_hypothesis
    return (valid_acc, total_loss, valid_ones, f1)
