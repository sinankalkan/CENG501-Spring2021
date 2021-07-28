import torchvision
import torch
import torch.nn as nn
from model.faster_rcnn import FasterRCNN


class D_fcn(nn.Module):
    def __init__(self, incnlsize = 256):
        super(D_fcn, self).__init__()
        self.conv1 = nn.Conv2d(incnlsize,int(incnlsize/2),kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv2 = nn.Conv2d(int(incnlsize/2),int(incnlsize/4),kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv3 = nn.Conv2d(int(incnlsize/4),1,kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x.view(-1,1)

class CAIACLSF(nn.Module):
    def __init__(self,n_classes = 2):
        super(CAIACLSF, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024,n_classes)

    def forward(self, x):
        x = self.relu(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x



class VGG16(FasterRCNN):
    def __init__(self, num_classes, class_agnostic, pretrained=False, model_path=None):
        FasterRCNN.__init__(self, num_classes, class_agnostic, 512)
        self.pretrained = pretrained
        self.model_path = model_path
        self.num_classes = num_classes

    def _init_modules(self):
        backbone = torchvision.models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s..." % (self.model_path))
            state_dict = torch.load(self.model_path)
            backbone.load_state_dict({k:v for k,v in state_dict.items() if k in backbone.state_dict()})
            print('Done.')
        
        backbone.classifier = nn.Sequential(*list(backbone.classifier._modules.values())[:-1])
        
        # not using the last maxpool layer
        self.RCNN_base1 = nn.Sequential(*list(backbone.features._modules.values())[:10])
        self.RCNN_base2 = nn.Sequential(*list(backbone.features._modules.values())[10:17])
        self.RCNN_base3 = nn.Sequential(*list(backbone.features._modules.values())[17:24])
        self.RCNN_base4 = nn.Sequential(*list(backbone.features._modules.values())[24:30])
        self.RCNN_base5 = nn.Sequential(*list(backbone.features._modules.values())[30:-1])

        self.D1_fcnn = D_fcn(incnlsize = 128)
        self.D2_fcnn = D_fcn(incnlsize = 256)
        self.D3_fcnn = D_fcn(incnlsize = 512)
        self.D4_fcnn = D_fcn(incnlsize = 512)

        self.caia_clsf = CAIACLSF(n_classes = self.num_classes)

        self.RCNN_top = backbone.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

        # Fix the layers before conv3 for VGG16:
        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    def _feed_pooled_feature_to_top(self, pooled_feature):
        return self.RCNN_top(pooled_feature.view(pooled_feature.size(0), -1))