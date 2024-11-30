# tests/test_batchnorm.py

import torch
from src.model import Net

def test_batchnorm():
    model = Net()
    bn_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    num_bn_layers = len(bn_layers)
    print(f"Number of BatchNorm2d layers: {num_bn_layers}")
    assert num_bn_layers > 0, "No BatchNorm2d layers found in the model!"

