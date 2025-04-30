import torch
from torchvision.models import resnet18, ResNet18_Weights

def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(512, 2)
    return model

def get_image_transform():
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess

if __name__ == "__main__":
    model = get_model()
    preprocess = get_image_transform()
    random_image = torch.rand((10, 3, 224, 224))
    print(model(preprocess(random_image)))
