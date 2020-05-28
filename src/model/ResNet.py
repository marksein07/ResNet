import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from inplace_abn import InPlaceABN


class ResLayer(nn.Module) :
    def __init__(self, num_features, bias = False) :
        super(ResLayer, self).__init__()
        self.add_module('Bottle_neck', nn.Sequential(
            nn.Conv2d(num_features, 64, 
                      kernel_size=1, stride = 1, padding = 0, bias = bias),
            InPlaceABN(64),
            nn.Conv2d(64, 64, 
                      kernel_size=3, stride = 1, padding = 1, bias = bias),
            InPlaceABN(64),
            nn.Conv2d(64, num_features, 
                      kernel_size=1, stride = 1, padding = 0, bias = bias), ) )
        self.add_module('ABN', InPlaceABN(num_features) )
    def forward(self, x) :
        return self.ABN(x+self.Bottle_neck(x))
    
class ResBlock(nn.ModuleDict) :
    def __init__(self,num_layers, num_features, bias = False ):
        super(ResBlock, self).__init__()
        for n_th_layer in range(num_layers) :
            layer = ResLayer(num_features, bias)
            self.add_module('ResLayers%d'%n_th_layer, layer)
    def forward(self, x) :
        for name, layer in self.items() :
            x = layer(x)
        return x
    
class ResNet(nn.Module) :
    def __init__(self, num_layers = (2,4,6,4), num_features = (64,128,256,512), bias = False) :
        super(ResNet, self).__init__()
        self.features_layers = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,num_features[0], 
                                                                    kernel_size=7, stride=1,
                                                                    padding = 3, bias = bias )),
                                                         ('ABN0', InPlaceABN(num_features[0]))]))
        self.features_layers.add_module('ResBlock0', ResBlock(num_layers[0], num_features[0]) )
        pre_num_feature = num_features[0]
        for i, (num_layer, num_feature) in enumerate(zip(num_layers[1:], num_features[1:]),1) :
            self.features_layers.add_module('conv%d'%i, nn.Conv2d(pre_num_feature,num_feature, 
                                                               kernel_size=3, stride=2,
                                                               padding = 1, bias = bias) )
            self.features_layers.add_module('ABN%d'%i, InPlaceABN(num_feature))
            #self.features_layers.add_module('pool%d'%i,     nn.AvgPool2d(2,2) )
        
            self.features_layers.add_module('ResBlock%d'%i, ResBlock(num_layer, num_feature) )
            pre_num_feature = num_feature
#         self.features_layers.add_module('conv2', nn.Conv2d(128, 256,
#                                                            kernel_size=3, stride=1,
#                                                            padding = 1, bias =False) )
#         self.features_layers.add_module('pool1',     nn.AvgPool2d(2,2) )
        
#         self.features_layers.add_module('ResBlock2', ResBlock(8, 256) )

#         self.features_layers.add_module('conv3', nn.Conv2d(256,512,
#                                                            kernel_size=3, stride=1,
#                                                            padding = 1, bias =False) )
#         self.features_layers.add_module('pool2',     nn.AvgPool2d(2,2) )
#         self.features_layers.add_module('ResBlock3', ResBlock(8, 512) )
        
        self.features_layers.add_module('GlobalPool', nn.AdaptiveAvgPool2d(1) )
        self.classifier = nn.Linear(num_features[-1],10,bias=False)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
    def forward(self, x) :
        features = self.features_layers(x)
        flatten_features = torch.flatten(features,1)
        logits = self.classifier(flatten_features)
        return logits
                     
        