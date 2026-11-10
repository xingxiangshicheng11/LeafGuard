from torch import nn
from torchvision import models


def build_resnet18(num_classes, pretrained=True, freeze_backbone=True, finetune_layer4=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
        if finetune_layer4:
            for p in model.layer4.parameters():
                p.requires_grad = True
    return model
