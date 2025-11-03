# ==============================
# train_model.py
# Digit Recognition AI - PyTorch Training Script
# Author: Abu Sufyan - Student
# Organization: Abu Zar
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ======================================================
# 1️⃣ Setup device (GPU if available)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ======================================================
# 2️⃣ Define CNN architecture
# ======================================================
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ======================================================
# 3️⃣ Load dataset (MNIST)
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ======================================================
# 4️⃣ Initialize model, loss, optimizer
# ======================================================
model = DigitCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================================================
# 5️⃣ Training function
# ======================================================
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

# ======================================================
# 6️⃣ Testing function
# ======================================================
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"📊 Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy

# ======================================================
# 7️⃣ Training loop
# ======================================================
EPOCHS = 10
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    acc = test()
    if acc > best_acc:
        best_acc = acc
        # Save best model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/digit_model.pth")
        print(f"💾 Saved new best model with accuracy: {best_acc:.2f}%\n")

print(f"🏁 Training complete! Best Accuracy: {best_acc:.2f}%")
