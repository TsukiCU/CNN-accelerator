import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
from torch.multiprocessing import freeze_support

# CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # The first convolutional module
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The second convolutional module
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The third convolutional module
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # FCN
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}], Step [{i + 1}/{len(trainloader)}], '
                  f'Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading datasets...")
    trainset = torchvision.datasets.CIFAR10(root='./basic_CNN/data', train=True,
                                          download=True, transform=transform_train)

    num_workers = min(4, multiprocessing.cpu_count())
    trainloader = DataLoader(trainset, batch_size=128,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True if torch.cuda.is_available() else False)

    testset = torchvision.datasets.CIFAR10(root='./basic_CNN/data', train=False,
                                         download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100,
                           shuffle=False, num_workers=num_workers,
                           pin_memory=True if torch.cuda.is_available() else False)

    print("Initializing model...")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_accuracy = 0
    print("Starting training...")

    for epoch in range(num_epochs):
        train(model, trainloader, criterion, optimizer, device, epoch)
        accuracy = test(model, testloader, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './basic_CNN/best_model.pth')
            print(f'Model saved. Highest accuracy : {best_accuracy:.2f}%')

    print(f'Finished training with accuracy : {best_accuracy:.2f}%')

if __name__ == '__main__':
    freeze_support()
    main()