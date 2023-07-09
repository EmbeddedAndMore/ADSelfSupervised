import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ADModel(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2):
        super(ADModel, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)


        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        
        # unpack layers to form a sequential model
        head = nn.Sequential(
            *sequential_layers
          )
        self.resnet18.fc = nn.Identity()
        self.head = head

        # output layer, without activation
        self.out = nn.Linear(last_layer, num_classes)
    
    def forward(self, x):
        embeds = self.resnet18(x)
        # apply head after resnet
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits
    
    def freeze_resnet(self):
        # freez full resnet18 if we loaded pretrained network
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
