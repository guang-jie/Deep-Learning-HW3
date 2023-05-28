import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
from torchsummary import summary

# Define LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

print(len(train_dataset), len(valid_dataset), len(test_dataset))

# Set batch size
batch_size = 32

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create LeNet-5 model
model = LeNet5().to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set number of epochs
num_epochs = 10

# Initialize lists to store training accuracy and validation accuracy
train_accuracy_list = []
valid_accuracy_list = []
test_accuracy_list = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_dataset)
    train_accuracy = 100.0 * train_correct / len(train_dataset)
    train_accuracy_list.append(train_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            valid_correct  += (predicted == labels).sum().item()
        
        valid_loss /= len(valid_dataset)
        valid_accuracy = 100.0 * valid_correct / len(valid_dataset)
        valid_accuracy_list.append(valid_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")



# Evaluation
model.eval()
test_loss = 0.0
test_correct = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
inference_time = time.time() - start_time
    
test_loss /= len(test_dataset)
test_accuracy = 100.0 * test_correct / len(test_dataset)
test_accuracy_list.append(test_accuracy)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print('inference_time:', inference_time)

# 輸出模型的參數量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters:", total_params)

# 使用 torchsummary 輸出模型的層次結構和記憶體使用量
summary(model, input_size=(1, 28, 28))

# Plot accuracy curve
plt.plot(range(1, num_epochs+1), train_accuracy_list, label='Train')
plt.plot(range(1, num_epochs+1), valid_accuracy_list, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
