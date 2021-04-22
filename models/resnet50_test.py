import timm
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=2)

    def forward(self, x):
        return self.model(x)
