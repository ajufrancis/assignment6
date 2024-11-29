import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# Set device to 'mps' if available, else fallback to 'cpu'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),  # Output: 16x28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        # CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # Output: 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        # Transition Block
        self.transblock1 = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False),  # Output: 16x28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 16x14x14
        )

        # CONV BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # Output: 32x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        # Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False),  # Output: 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 16x7x7
        )

        # CONV BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # Output: 32x7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        # OUTPUT BLOCK
        self.output = nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False),  # Output: 10x7x7
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transblock1(x)
        x = self.convblock3(x)
        x = self.transblock2(x)
        x = self.convblock4(x)
        x = self.output(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        log_probs = F.log_softmax(outputs, dim=-1)
        targets = F.one_hot(targets, num_classes=10).float()
        targets = targets * self.confidence + self.smoothing / 10
        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss

def train(model, device, train_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader, ncols=100)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        accuracy = 100 * correct / processed
        pbar.set_description(f'Epoch: {epoch} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc_value = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc_value:.2f}%)\n')
    return test_loss, test_acc_value

def main():
    torch.manual_seed(1)
    batch_size = 128  # Increased batch size

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation((-8.0, 8.0), fill=(0,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = Net().to(device)
    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.05)  # Reduced smoothing
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos'
    )

    num_epochs = 20
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, scheduler, epoch)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        if test_accuracy >= 99.4:
            print(f"Reached 99.4% accuracy at epoch {epoch}!")
            break

    print(f"Best accuracy achieved: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()
