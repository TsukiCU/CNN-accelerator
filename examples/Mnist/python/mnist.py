import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The same parameters as our model.
class MnistNN(nn.Module):
    def __init__(self):
        super(MnistNN, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

learning_rate = 0.05
num_epochs = 5
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(), # Flatten to tensor.
    transforms.Normalize((0.5,), (0.5,))
])

################## Training ##################

train_dataset = datasets.MNIST(
    root='.', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    root='.', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = MnistNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"\033[1;36m\nEpoch {epoch + 1} training starts.\033[0m\n")
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 150 == 0:
            print(f"Epoch {epoch + 1} Batch {batch_idx}, Loss: {total_loss / (batch_idx + 1):.4f}")

################## Testing ##################
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")