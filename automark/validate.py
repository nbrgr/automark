import torch

def validate_on_data(batch_size = 1, data=None, eval_metric=None,
        model=None, use_cuda=False, loss_function=None):
    
    #Return valid_score, valid_loss, valid_sources, valid_references, valid_hypothesis
    return (None, None, None, None, None)