import os
import sys
from collections import defaultdict

import torch

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from transformers import BertTokenizer


def tensorify(fun, dtype):
    def tensorified_fun(batch):
        return torch.tensor(fun(batch), dtype=dtype)
    return tensorified_fun


def batch_fun(batch):
    max_len = max([len(x) for x in batch])

    for example in batch:
        example += [0.0] * (max_len - len(example))

    return batch


def identity_fun(batch):
    return batch


def make_dataset(config):
    bert_path = config['bert']['path']
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    unk_id = tokenizer.vocab['[UNK]']
    pad_id = tokenizer.vocab['[PAD]']
    unk_fun = lambda: unk_id

    vocab = defaultdict(unk_fun)

    for k,v in tokenizer.vocab.items():
        vocab[k] = v

    # load data from files
    src_ext = config['data']['source']
    trg_ext = config['data']['target']
    ma_ext = config['data']['marking']
    train_path = config['data']['train']
    dev_path = config['data']['dev']
    bad_weight = config['train'].get('bad_weight', 1.0)

    src_trg_field = data.Field(eos_token=None,
                           pad_token=pad_id, 
                           batch_first=True,
                           include_lengths=True,
                           sequential=True,
                           use_vocab=False)

    ann_field = data.RawField(postprocessing=tensorify(batch_fun, torch.long))

    mask_field = data.RawField(postprocessing=tensorify(batch_fun,
                                                        torch.float32))

    loss_weight_field = data.RawField(postprocessing=tensorify(batch_fun,
                                                        torch.float32))

    id_mask = data.RawField(postprocessing=tensorify(batch_fun, torch.long))

    train_data = MergeDataset(path=train_path,
                              exts=(src_ext, trg_ext, ma_ext),
                              fields=(src_trg_field, ann_field,
                                      mask_field, id_mask, loss_weight_field),
                              bos_token='[CLS]', bad_weight=bad_weight,
                              sep_token='[SEP]', vocab=vocab)

    dev_data = MergeDataset(path=dev_path,
                            exts=(src_ext, trg_ext, ma_ext),
                            fields=(src_trg_field, ann_field, mask_field,
                                    id_mask, loss_weight_field), vocab=vocab,
                            bos_token='[CLS]', sep_token='[SEP]')
    test_data = None
    """
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    """
    return train_data, dev_data, test_data, 


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    if train:
        # optionally shuffle and sort during training
        data_iter = data.Iterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src_trg), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.Iterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size,
            train=False, sort=False)

    return data_iter


class MergeDataset(Dataset):

    def __init__(self, path, exts, fields, bos_token = '[CLS]',
                 sep_token = '[SEP]', vocab=None, bad_weight=1.0, **kwargs):

        src_len = data.RawField(
            postprocessing=tensorify(identity_fun, torch.long))
        trg_len = data.RawField(
            postprocessing=tensorify(identity_fun, torch.long))
        attention_mask = data.RawField(postprocessing=tensorify(batch_fun, torch.long))
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src_trg', fields[0]), ('weights', fields[1]),
                      ('label_mask', fields[2]), ('id_mask', fields[3]),
                      ('loss_weight', fields[4]),
                      ('src_len', src_len), ('trg_len', trg_len),
                      ('attention_mask', attention_mask)]

        src_path, trg_path, weights_path = tuple(
            os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file, \
                open(weights_path) as weight_file:
            for src_line, trg_line, weights_line in \
                    zip(src_file, trg_file, weight_file):
                src_line, trg_line = src_line.strip().split(" "), \
                                     trg_line.strip().split(" ")
                
                src_line = [vocab[bos_token]] + [vocab[x] for x in src_line] +\
                           [vocab[sep_token]]
                trg_line = [vocab[x] for x in trg_line]
                label_mask = [0.0] * len(src_line) + [1.0] * len(trg_line)
                trg_weights = [int(w) for w in weights_line.strip().split(" ")]
                weights = [0] * (len(src_line)) + trg_weights
                # give higher weight for 0 labels
                loss_weights = [0.0] * (len(src_line)) + \
                               [1.0 if w == 1 else bad_weight
                                for w in trg_weights]

                if src_line != '' and trg_line != '':
                    merged_line = src_line + trg_line
                    att_mask = [1] * len(merged_line)
                    id_mask = [0] * len(src_line) + [1] * len(trg_line)
                    assert len(merged_line) == len(id_mask)
                    assert len(label_mask) == len(merged_line)
                    assert len(weights) == len(merged_line)
                    assert len(loss_weights) == len(merged_line)
                    examples.append(data.Example.fromlist(
                        [merged_line, weights, label_mask, id_mask,
                         loss_weights, len(src_line), len(trg_line), att_mask], fields))

        super(MergeDataset, self).__init__(examples, fields, **kwargs)