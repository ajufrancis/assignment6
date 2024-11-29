# tests/test_dropout.py

import torch
from src.model import Net

def main():
    model = Net()
    dropout_layers = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    num_dropout_layers = len(dropout_layers)
    print(f"Number of Dropout layers: {num_dropout_layers}")
    if num_dropout_layers == 0:
        raise Exception("No Dropout layers found in the model!")
    else:
        print("Dropout test passed.")

if __name__ == "__main__":
    main()

