from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field


#This class is taken from Joey
class WeightedTranslationDataset(Dataset):
    """ Defines a parallel dataset with weights for the targets. """

    def __init__(self, path, exts, fields, level, feedback_weights, **kwargs):
        """Create a TranslationDataset given paths and fields.
        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        :param level: char or word or bpe
        :param feedback_weights: dictionary mapping feedback values to
          loss weights
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]),
                      ('weights', fields[2])]

        src_path, trg_path, feedback_path = tuple(os.path.expanduser(path + x)
                                                  for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file, \
                open(feedback_path) as feedback_file:
            for src_line, trg_line, weights_line in \
                    zip(src_file, trg_file, feedback_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                weights = [float(weight) for weight in
                           weights_line.strip().split(" ")]
                if feedback_weights is not None:
                    # lookup weights for individual feedback types
                    weights = [feedback_weights.get(w, w) for w in weights]
                if src_line != '' and trg_line != '':
                    # there must be feedback for every token
                    if level == "char":
                        char_weights = []
                        # distribute feedback from tokens over chars
                        assert len(trg_line.split()) == len(weights)
                        for trg_token, token_weight in zip(trg_line.split(),
                                                           weights):
                            # replicate weight for every char in trg token
                            # and for following whitespace
                            char_weights.extend(
                                (len(trg_token)+1)*[token_weight])
                        # remove last added weight for whitespace
                        weights = char_weights[:-1]
                    if len(weights) == 1 and "sentence" in feedback_path: 
                        # one score for the full sentence
                        weights = [weights[0]]*len(fields[1][1].tokenize(trg_line))
                    assert len(weights) == len(fields[1][1].tokenize(trg_line))
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, weights], fields))

        super(WeightedTranslationDataset, self).__init__(examples,
                                                         fields, **kwargs)