from torch import nn
from lib.models import extracter
from lib.base.base_model import BaseModel


class Encoder(BaseModel):
    def __init__(self, in_channels, backbone='resnet50', pretrained=True):
        super().__init__()
        model = getattr(extracter, backbone)(pretrained)

        if in_channels == 4:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.stride2_features = self.layer1[0].conv1.in_channels
        self.stride4_features = self.layer2[0].conv1.in_channels
        self.stride8_features = self.layer4[2].conv3.out_channels

    def forward(self, x):
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x_1 = self.relu(x)
        x = self.maxpool(x_1)  # /2

        x_2 = self.layer1(x)
        x = self.layer2(x_2)   # /2
        x = self.layer3(x)
        x = self.layer4(x)

        return x_1, x_2, x
