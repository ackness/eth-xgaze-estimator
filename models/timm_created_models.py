import timm
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(Model, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=2)

    def forward(self, x):
        return self.model(x)
