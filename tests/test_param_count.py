# tests/test_param_count.py

import torch
from src.model import Net  # Adjust the path if necessary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    model = Net()
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params}")
    if total_params > 20000:
        raise Exception(f"Total parameters exceed 20k! ({total_params})")
    else:
        print("Parameter count test passed.")

if __name__ == "__main__":
    main()

