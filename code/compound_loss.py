import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision import models


class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats


class CompoundLoss(_Loss):

    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=1, resnet_weight=0.01):
        super(CompoundLoss, self).__init__()

        self.mse_weight = mse_weight
        self.resnet_weight = resnet_weight

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0

        input_feats = self.model(torch.cat([input, input, input], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss = self.mse_weight * self.criterion(input, target) + self.resnet_weight * loss_value

        return loss
