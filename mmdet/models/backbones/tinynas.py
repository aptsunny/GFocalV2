# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import os

import torch
#需要在 masternet 前添加 . 表示从当前目录导入
from .masternet import MasterNet
################################
#                              #
#        定义 TinyNAS 类        #
#                              #
################################
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

@BACKBONES.register_module
class TinyNAS(nn.Module):
    def __init__(self, net_str=None):
        super(TinyNAS, self).__init__()
        self.body, _ = get_backbone(
            net_str,
            load_weight=False,
            task='detection')
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
    def forward(self, x):
        """Forward function."""
        return self.body(x)
################################
#                              #
#     加载搜索到的 backbone      #
#                              #
################################
def get_backbone(filename,
                 load_weight=True,
                 network_id=0,
                 task='classification'):
    # load best structures
    with open(filename, 'r') as fin:
        content = fin.read()
        output_structures = ast.literal_eval(content)

    network_arch = output_structures['space_arch']
    best_structures = output_structures['best_structures']

    # If task type is classification, param num_classes is required
    out_indices = (1, 2, 3, 4) if task == 'detection' else (4, )
    backbone = MasterNet(
            structure_info=best_structures[network_id],
            out_indices=out_indices,
            num_classes=1000,
            task=task)

    return backbone, network_arch
################################
#                              #
#         测试前向推理           #
#                              #
################################
if __name__ == '__main__':
    # make input
    x = torch.randn(1, 3, 224, 224)

    # instantiation
    backbone, network_arch = get_backbone('best_structure.json')

    print(backbone)
    # forward
    input_data = [x] 
    pred = backbone(*input_data)
    
    #print output
    for o in pred:
        print(o.size())
