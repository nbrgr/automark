import random

en = open('dummy.en', 'w')
hyp = open('dummy.hyp', 'w')
ann = open('dummy.ann', 'w')

en_vocab = ['cat', 'dog']
hyp_vocab = ['katze', 'hund']
ann_vocab = [0, 1]

size = 128
length = 10

for i in range(size):
    choices = [random.choice(ann_vocab) for j in range(length)]

    en_string = " ".join([en_vocab[j] for j in choices])
    hyp_string = " ".join([hyp_vocab[j] for j in choices])
    ann_string = " ".join([str(choice) for choice in choices])

    en.write(en_string + "\n")
    hyp.write(hyp_string + "\n")
    ann.write(ann_string + "\n")

en.close()
hyp.close()
ann.close()