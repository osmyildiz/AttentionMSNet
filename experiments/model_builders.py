"""
Shared model builders for all steps.
"""
import torch.nn as nn
import torchvision.models as models


def build_efficientnet_b3(num_classes=3, dropout=0.4):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1536, num_classes))
    return model

def build_densenet169(num_classes=3, dropout=0.4):
    model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1664, num_classes))
    return model

def build_resnet50(num_classes=3, dropout=0.4):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    return model

BASELINE_BUILDERS = {
    "EfficientNet-B3": build_efficientnet_b3,
    "DenseNet-169": build_densenet169,
    "ResNet-50": build_resnet50,
}
