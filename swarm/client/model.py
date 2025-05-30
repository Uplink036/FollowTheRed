import torch
from torchvision.models import resnet18
from torchvision import transforms as T


def get_model(amount_of_colours=1):
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 2*amount_of_colours),
        torch.nn.Hardtanh()
    )
    return model


def get_image_transform():
    preprocess = T.Compose(
        [T.Resize(256),
         T.CenterCrop(224)]
    )
    return preprocess

if __name__ == "__main__":
    model = get_model()
    preprocess = get_image_transform()
    random_image = torch.rand((10, 3, 224, 224))
    print(model(preprocess(random_image)))

