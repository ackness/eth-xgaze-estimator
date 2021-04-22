import torch
import torch.nn.functional as F


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.metrics = lambda x, y: 1 - torch.mean(F.cosine_similarity(x, y, dim=-1))

    def forward(self, x, label):
        return self.metrics(x, label)
