import torch

def load_weights(model, weights):
    """
    Loads the weights of only the layers present in the given model.
    Ucitava samo tezine slojeva koji postoje u modelu, ako patch weights ide u whole image model, model ima dodatne PHRefiner blokove a pretrained weights nemaju 
    """
    # Ucitavanje pretreniranih 
    pretrained_dict = torch.load(weights, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
