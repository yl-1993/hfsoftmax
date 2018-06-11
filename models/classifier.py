import torch.nn as nn

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
