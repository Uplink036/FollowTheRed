import torch
import copy
from collections import OrderedDict

def agglomerate(model_weights: list[torch.nn.Module]) -> OrderedDict:
    avg_weight = copy.deepcopy(model_weights[0])    
    for key in avg_weight.keys():
        for weight in model_weights[1:]:
            avg_weight[key] += weight[key]
        avg_weight[key] /= len(model_weights)        
    return avg_weight
    