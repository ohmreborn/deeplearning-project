import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.features.children())[:36]).eval()
        for p in self.features.parameters(): 
            p.requires_grad = False
    def forward(self, x) -> torch.Tensor:
        return self.features(x)
                                                                        
