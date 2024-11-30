# MNIST Classification with Custom CNN Architecture

## Introduction

This project implements a custom Convolutional Neural Network (CNN) to classify images from the MNIST dataset. The goal is to achieve a validation/test accuracy of 99.4% within 20 epochs, while adhering to specific architectural constraints, such as using less than 20,000 parameters, incorporating Batch Normalization, Dropout, and Global Average Pooling (GAP) instead of fully connected layers.

## Requirements

- Python 3.7 or higher
- torch
- torchvision
- pytest
- tqdm
- numpy

All dependencies are listed in the requirements.txt file.

# Project Structure

```
Assignment6/
├── .github/
│   └── workflows/
│       └── model_checks.yml     # GitHub Actions workflow file
├── src/
│   ├── __init__.py
│   └── model.py                 # Model definition and training script
├── tests/
│   ├── __init__.py
│   ├── test_param_count.py      # Test for parameter count
│   ├── test_batchnorm.py        # Test for Batch Normalization usage
│   ├── test_dropout.py          # Test for Dropout usage
│   └── test_gap_fc.py           # Test for GAP or FC layer usage
├── requirements.txt             # List of project dependencies
├── README.md                    # Project documentation
└── other_files.py               # Additional scripts (if any)
```

# Installation

1. Clone the repository:
 ```
 git clone https://github.com/ajufrancis/Assignment6.git
 cd Assignment6
 ```
2. Set up a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

# Usage

## Training the Model
To train the model on the MNIST dataset, run:
```
python src/model.py --epochs 20 --batch-size 128
```
Command-Line Arguments:

- --epochs: Number of epochs to train (default: 20)
- --batch-size: Batch size for training (default: 128)

## Running Tests

To run the test suite and verify the model against the specified constraints, execute:
```
pytest -s
```

# Model Architecture

## Layer Details

The model architecture consists of the following blocks:

1. **Input Block**
    - Conv2d(1, 16, 3, padding=1, bias=False)
    - BatchNorm2d(16)
    - ReLU
    - Dropout(0.05)
2. **Convolutional Block 1**
    - Conv2d(16, 32, 3, padding=1, bias=False)
    - BatchNorm2d(32)
    - ReLU
    - Dropout(0.05)
3. **Transition Block 1**
    - Conv2d(32, 16, 1, bias=False)
    - BatchNorm2d(16)
    - ReLU
    - MaxPool2d(2, 2)
4. **Convolutional Block 2**
    - Conv2d(16, 32, 3, padding=1, bias=False)
    - BatchNorm2d(32)
    - ReLU
    - Dropout(0.05)
5. **Transition Block 2**
    - Conv2d(32, 16, 1, bias=False)
    - BatchNorm2d(16)
    - ReLU
    - MaxPool2d(2, 2)
7. **Convolutional Block 3**
    - Conv2d(16, 32, 3, padding=1, bias=False)
    - BatchNorm2d(32)
    - ReLU
    - Dropout(0.05)
    - Output Block
    - Conv2d(32, 10, 1, bias=False)
    - BatchNorm2d(10)
    - ReLU
    - AdaptiveAvgPool2d(1)

**Parameter Calculation**
Total parameters: **15,620**

## Detailed Calculation:

- Input Block: 176 parameters
- Convolutional Block 1: 4,672 parameters
- Transition Block 1: 544 parameters
- Convolutional Block 2: 4,672 parameters
- Transition Block 2: 544 parameters
- Convolutional Block 3: 4,672 parameters
- Output Block: 340 parameters
- Sum: 176 + 4,672 + 544 + 4,672 + 544 + 4,672 + 340 = 15,620

# Results

After training the model, it achieves the desired accuracy on the MNIST dataset within the specified constraints.

**Accuracy Verification Log**
```
 % python3 model.py
Using device: mps
Epoch: 1 | Loss: 0.9313 | Accuracy: 78.01%: 100%|█████████████████| 469/469 [00:18<00:00, 24.96it/s]

Test set: Average loss: 0.7802, Accuracy: 9516/10000 (95.16%)

Epoch: 2 | Loss: 0.4514 | Accuracy: 95.66%: 100%|█████████████████| 469/469 [00:17<00:00, 27.11it/s]

Test set: Average loss: 0.4165, Accuracy: 9778/10000 (97.78%)

Epoch: 3 | Loss: 0.4213 | Accuracy: 96.96%: 100%|█████████████████| 469/469 [00:17<00:00, 26.59it/s]

Test set: Average loss: 0.3844, Accuracy: 9798/10000 (97.98%)

Epoch: 4 | Loss: 0.3681 | Accuracy: 97.53%: 100%|█████████████████| 469/469 [00:17<00:00, 26.22it/s]

Test set: Average loss: 0.3679, Accuracy: 9841/10000 (98.41%)

Epoch: 5 | Loss: 0.4200 | Accuracy: 97.86%: 100%|█████████████████| 469/469 [00:17<00:00, 26.38it/s]

Test set: Average loss: 0.3914, Accuracy: 9850/10000 (98.50%)

Epoch: 6 | Loss: 0.3874 | Accuracy: 98.06%: 100%|█████████████████| 469/469 [00:17<00:00, 26.42it/s]

Test set: Average loss: 0.4243, Accuracy: 9718/10000 (97.18%)

Epoch: 7 | Loss: 0.3470 | Accuracy: 98.20%: 100%|█████████████████| 469/469 [00:17<00:00, 26.24it/s]

Test set: Average loss: 0.3770, Accuracy: 9815/10000 (98.15%)

Epoch: 8 | Loss: 0.3837 | Accuracy: 98.20%: 100%|█████████████████| 469/469 [00:17<00:00, 26.30it/s]

Test set: Average loss: 0.3590, Accuracy: 9871/10000 (98.71%)

Epoch: 9 | Loss: 0.3751 | Accuracy: 98.29%: 100%|█████████████████| 469/469 [00:18<00:00, 26.03it/s]

Test set: Average loss: 0.3694, Accuracy: 9832/10000 (98.32%)

Epoch: 10 | Loss: 0.3363 | Accuracy: 98.34%: 100%|████████████████| 469/469 [00:17<00:00, 26.29it/s]

Test set: Average loss: 0.3532, Accuracy: 9880/10000 (98.80%)

Epoch: 11 | Loss: 0.3579 | Accuracy: 98.41%: 100%|████████████████| 469/469 [00:18<00:00, 25.65it/s]

Test set: Average loss: 0.3525, Accuracy: 9895/10000 (98.95%)

Epoch: 12 | Loss: 0.3249 | Accuracy: 98.53%: 100%|████████████████| 469/469 [00:17<00:00, 26.14it/s]

Test set: Average loss: 0.3455, Accuracy: 9903/10000 (99.03%)

Epoch: 13 | Loss: 0.3394 | Accuracy: 98.55%: 100%|████████████████| 469/469 [00:17<00:00, 26.12it/s]

Test set: Average loss: 0.3467, Accuracy: 9904/10000 (99.04%)

Epoch: 14 | Loss: 0.3773 | Accuracy: 98.69%: 100%|████████████████| 469/469 [00:17<00:00, 26.11it/s]

Test set: Average loss: 0.3434, Accuracy: 9877/10000 (98.77%)

Epoch: 15 | Loss: 0.3535 | Accuracy: 98.80%: 100%|████████████████| 469/469 [00:18<00:00, 26.05it/s]

Test set: Average loss: 0.3293, Accuracy: 9909/10000 (99.09%)

Epoch: 16 | Loss: 0.3469 | Accuracy: 98.80%: 100%|████████████████| 469/469 [00:18<00:00, 25.56it/s]

Test set: Average loss: 0.3258, Accuracy: 9918/10000 (99.18%)

Epoch: 17 | Loss: 0.3601 | Accuracy: 98.92%: 100%|████████████████| 469/469 [00:18<00:00, 25.99it/s]

Test set: Average loss: 0.3222, Accuracy: 9937/10000 (99.37%)

Epoch: 18 | Loss: 0.3413 | Accuracy: 98.99%: 100%|████████████████| 469/469 [00:18<00:00, 26.05it/s]

Test set: Average loss: 0.3188, Accuracy: 9939/10000 (99.39%)

Epoch: 19 | Loss: 0.3356 | Accuracy: 99.10%: 100%|████████████████| 469/469 [00:17<00:00, 26.16it/s]

Test set: Average loss: 0.3172, Accuracy: 9943/10000 (99.43%)

Reached 99.4% accuracy at epoch 19!
Best accuracy achieved: 99.43%
```




