import torch
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
resnet18_model = resnet18(weights=weights)

torch.save(resnet18_model.state_dict(), "resnet18.pth")

for name, param in resnet18_model.state_dict().items():
    print(name, "\t", param.size())

