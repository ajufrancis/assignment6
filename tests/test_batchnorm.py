# tests/test_batchnorm.py

import torch
from src.model import Net

def main():
    model = Net()
    bn_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    num_bn_layers = len(bn_layers)
    print(f"Number of BatchNorm2d layers: {num_bn_layers}")
    if num_bn_layers == 0:
        raise Exception("No BatchNorm2d layers found in the model!")
    else:
        print("Batch Normalization test passed.")

if __name__ == "__main__":
    main()

