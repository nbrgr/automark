import os

from collections import defaultdict

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from transformers import BertTokenizer

def make_dataset(config, bert_path):

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    unk_id = tokenizer.vocab['[UNK]']
    init_id = tokenizer.vocab['[CLS]']
    pad_id = tokenizer.vocab['[PAD]']
    unk_fun = lambda: unk_id

    vocab = defaultdict(unk_fun)

    for k,v in tokenizer.vocab.items():
        vocab[k] = v

    # load data from files
    src_ext = config['data']['source']
    trg_ext = config['data']['target']
    ma_ext = config['data']['marking']
    train_path = config["train"]
    dev_path = config["dev"]
    max_sent_length = config["max_sent_length"]


    src_trg_field = data.Field(init_token=init_id, eos_token=None,
                           pad_token=pad_id, 
                           batch_first=True,
                           include_lengths=True,
                           sequential=True,
                           use_vocab=False)

    def batch_fun(batch):
        max_len = max([len(x) for x in batch])

        for example in batch:
            example += [0.0] * (max_len - len(example))

        return batch

    ann_field = data.RawField(postprocessing=batch_fun)

    train_data = MergeDataset(path=train_path,
                                    exts=(src_ext, trg_ext, ma_ext),
                                    fields=(src_trg_field, ann_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length,
                                    sep_token='[SEP]')

    dev_data = MergeDataset(path=dev_path,
                                  exts=(src_ext, trg_ext, ma_ext),
                                  fields=(src_trg_field, ann_field))
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


class MergeDataset(Dataset):
    def __init__(self, path, exts, fields, sep_token = '[SEP]', vocab=None, **kwargs):
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src_trg', fields[0]), ('weights', fields[1])]

        src_path, trg_path, weights_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file, \
            open(weights_path) as weight_file:
            for src_line, trg_line, weights_line in \
                zip(src_file, trg_file, weight_file):
                src_line, trg_line = src_line.strip().split(" "), trg_line.strip().split(" ")
                
                src_line = [vocab[x] for x in src_line]
                trg_line = [vocab[x] for x in trg_line]

                weights = [float(weight) for weight in weights_line.strip().split(" ")]
                weights = [0.0] * (len(src_line) + 1) + weights
                
                if src_line != '' and trg_line != '':
                    merged_line = src_line + [vocab[sep_token]] + trg_line
                    examples.append(data.Example.fromlist([merged_line, weights], fields))

        super(MergeDataset, self).__init__(examples, fields, **kwargs)