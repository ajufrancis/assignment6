# tests/test_param_count.py

import unittest
import torch

from src.model import Net  # Relative import within the package

class TestParameterCount(unittest.TestCase):
    def test_param_count(self):
        model = Net()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
        self.assertTrue(total_params <= 20000, f"Total parameters exceed 20k! ({total_params})")

if __name__ == "__main__":
    unittest.main()

