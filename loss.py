# Check this gist -> https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        
        vgg = torchvision.models.vgg16(pretrained=True)
        
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        
        for block in blocks:
            for param in block:
                param.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        
        self.transform = nn.functional.interpolate
        
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        self.resize = resize

    def forward(self, input, target):
        input = input.cpu()
        target = target.cpu()
        
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.l1_loss(x, y)
            
        return loss
