import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet, DenseNet121_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=3): 
        super(FishClassifier, self).__init__()
        self.desnet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.desnet.classifier = nn.Sequential(
            nn.Linear(self.desnet.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        return self.desnet(x)