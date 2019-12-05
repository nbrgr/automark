import torch
from torch.nn import Functional as F

from automark.mark import AutoMark
from automark.dataset import make_dataset

class Trainer:
    def __init__(self, model, dataset, epochs, optimizer):

        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = optimizer

    def train(self):
        for i in range(self.epochs):
            self.run_epoch()

    def run_epoch(self, iterator):
        for sentences, markings in iterator:
            predictions = self.model(sentences)
            loss = self.loss(predictions, markings)
            loss.backwards()
            self.optimizer.step()

    def loss(self, predictions, markings):
        return F.nll_loss(predictions, markings, dim=-1)

def trainer_builder(config):
    model_path = config['bert']['path']
    model = AutoMark(model=model_path)
    model.train()
    dataset = make_dataset(config, model_path)

    epochs = config['train']['epochs']

    optimizer_str = config['train']['optimizer']
    lr = config['train']['lr']

    if optimizer_str == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = Trainer(model, dataset, epochs, optimizer)
    trainer.train()