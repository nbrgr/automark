import numpy as np

import torch
from torch.nn import functional as F

from transformers import BertTokenizer

import os

from automark.helpers import load_config


def preprocess(config_path):
    config = load_config(config_path)
    print("Read Config {}".format(config_path))
    pretrained_path = config['bert']['path']

    source_postfix = config['data']['source']
    target_postfix = config['data']['target']
    marking_postfix = config['data']['marking']

    raw_train_files = config['data']['raw_train']
    train_files = config['data']['train']

    raw_dev_files = config['data']['raw_dev']
    dev_files = config['data']['dev']

    for input_files, output_files in [(raw_train_files, train_files),
                                      (raw_dev_files, dev_files)]:

        tokenizer = BertTokenizer.from_pretrained(pretrained_path)

        source_tokenfile = open(output_files + source_postfix, 'w')
        target_tokenfile = open(output_files + target_postfix, 'w')
        marking_distfile = open(output_files + marking_postfix, 'w')

        with open(input_files + source_postfix) as source_file, \
            open(input_files + target_postfix) as target_file, \
            open(input_files + marking_postfix) as marking_file:
            for source_line, target_line, marking_line in \
                    zip(source_file, target_file, marking_file):
                source = source_line.strip().split(" ")
                target = target_line.strip().split(" ")
                marking = marking_line.strip().split(" ")

                src_tokens = []
                trg_tokens = []
                marking_dist = []

                for word in source:
                    src_tokens.extend(
                        tokenizer.wordpiece_tokenizer.tokenize(word))

                for word, mark in zip(target, marking):
                    tokens = tokenizer.wordpiece_tokenizer.tokenize(word)
                    trg_tokens.extend(tokens)
                    marking_dist.extend([mark] * len(tokens))

                src_tok_string = " ".join(src_tokens)
                trg_tok_string = " ".join(trg_tokens)
                mark_dist_string = " ".join([str(x) for x in marking_dist])
                source_tokenfile.write(src_tok_string + "\n")
                target_tokenfile.write(trg_tok_string + "\n")
                marking_distfile.write(mark_dist_string + '\n')
        source_tokenfile.close()
        target_tokenfile.close()
        marking_distfile.close()

        print("Wrote files to {}[{}|{}|{}].".format(
            train_files, source_postfix, target_postfix, marking_postfix))
    print("Tokenization complete.")
