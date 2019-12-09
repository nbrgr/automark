import torch
from automark.dataset import make_data_iter, batch_to

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
        batch_count = 0
        for i, valid_batch in enumerate(valid_iter):
            batch_count += 1
            if use_cuda:
                valid_batch = batch_to(valid_batch, "cuda")
            batch_loss, ones, acc, predictions = model.get_loss_for_batch(
                valid_batch, loss_function, None
            )

            pred_labels = predictions.argmax(-1).view(-1).cpu().numpy()
            labels = valid_batch.weights.view(-1).cpu().numpy()
            label_mask = valid_batch.id_mask.view(-1).cpu().numpy()
            pred_labels_masked = pred_labels[label_mask == 1]
            labels_masked = labels[label_mask == 1]
            pred_label_list.extend(pred_labels_masked.tolist())
            label_list.extend(labels_masked.tolist())
            total_loss += batch_loss.item()
            total_tokens += valid_batch.trg_len.sum().item()
            total_seqs += valid_batch.src_trg[0].shape[0]
            valid_ones += ones
            valid_acc += acc

        valid_ones = valid_ones / batch_count
        valid_acc = valid_acc / total_seqs
        total_loss = total_loss / total_tokens

    f1 = f1_score(label_list, pred_label_list, average=None)

    if eval_metric == "f1_prod":
        valid_score = f1[0]*f1[1]
    elif eval_metric == "f1_0":
        valid_score = f1[0]
    elif eval_metric == "f1_1":
        valid_score = f1[1]
    elif eval_metric == "acc":
        valid_score = valid_acc
    else:
        raise ValueError("Please specify valid eval metric "
                         "[f1_prod, f1_0, f1_1, acc]")
    return valid_score, total_loss, valid_ones, f1
