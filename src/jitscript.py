from torch import jit
from model.DenseNet import DenseNet
import torch
import torchvision
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
class DenseNetJIT(jit.ScriptModule):
    def __init__(self):
        super(DenseNetJIT,self).__init__()
        self.add_module('densenet', jit.trace(DenseNet( growth_rate=12, DenseBlock_layer_num=(40,40,40), 
                     bottle_neck_size=4, dropout_rate=0.2, compression_rate=0.5, num_init_features=16,
                     num_input_features=3, num_classes=10, bias=False, memory_efficient=False),
                                              torch.rand(1,3,32,32) ) )
    def forward(self,x):
        return self.densenet(x)
    
model = DenseNetJIT().cuda()

BATCH_SIZE=12

transform_test  = transforms.Compose(
    [transforms.ToTensor(), ])

testset = torchvision.datasets.CIFAR10(root='./mnist', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


dataiter = iter(testloader)
images, labels = dataiter.next()  

out = model(images.cuda())
print(out)
                        