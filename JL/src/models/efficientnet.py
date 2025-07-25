import torch
import torch.nn as nn
import timm

class EfficientNet(nn.Module):
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        print("create model efficientNet")
        self.backbone = timm.create_model(cfg.model.model_name,
                                          pretrained=cfg.model.pretrained,
                                          num_classes=cfg.model.num_classes)
        
        for name, param in self.backbone.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True