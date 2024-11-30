# tests/test_gap_fc.py

import torch
from src.model import Net

def test_gap_fc():
    model = Net()
    fc_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    gap_layers = [module for module in model.modules() if isinstance(module, torch.nn.AdaptiveAvgPool2d)]

    num_fc_layers = len(fc_layers)
    num_gap_layers = len(gap_layers)

    print(f"Number of Fully Connected (Linear) layers: {num_fc_layers}")
    print(f"Number of Global Average Pooling layers: {num_gap_layers}")

    assert num_fc_layers > 0 or num_gap_layers > 0, "Neither Fully Connected layers nor GAP layers found in the model!"

