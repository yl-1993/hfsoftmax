import torch
import torch.nn as nn
from .ext_layers import HFSampler, HNSWSampler, Ident


class Classifier(nn.Module):
    def __init__(self, base, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.logits = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)
        x = self.logits(x)
        return x


def var_hook(grad):
    # hook can be used to send grad to ParameterServer
    return grad


class HFClassifier(nn.Module):
    def __init__(self, base, rank, feature_dim, sampler_num, num_classes):
        super(HFClassifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.hf_sampler = HFSampler(rank, feature_dim, sampler_num, num_classes)

    def forward(self, x, labels):
        x = self.base(x)
        x = self.dropout(x)
        w, b, labels = self.hf_sampler(x, labels)
        labels = labels.detach()
        # w.register_hook(var_hook)
        # asssert w.requires_grad == True
        x = torch.mm(x, w.t()) + b
        return x, labels


class HNSWClassifier(nn.Module):
    def __init__(self, base, rank, feature_dim, sampler_num, num_classes):
        super(HNSWClassifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.hnsw_sampler = HNSWSampler(rank, feature_dim, sampler_num, num_classes)

    def forward(self, x, labels):
        x = self.base(x)
        x = self.dropout(x)
        w, b, labels = self.hnsw_sampler(x, labels)
        labels = labels.detach()
        x = torch.mm(x, w.t()) + b
        return x, labels
