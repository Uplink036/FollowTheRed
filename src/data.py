import torch
from torchvision.models import resnet18, ResNet18_Weights
from dataclasses import dataclass


@dataclass
class jetbotAction:
    forward: float
    turn: float

@dataclass
class modelStruct:
    model: resnet18
    loss: torch.nn.MSELoss
    optimizer: torch.optim.Adam
    preprocess: ResNet18_Weights