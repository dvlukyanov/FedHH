from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Initialize the Accelerator
accelerator = Accelerator()

# Define the transforms for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
])

# Load CIFAR-10 data
train_dataset = datasets.CIFAR10(root='./ddp/data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./ddp/data', train=False, download=True, transform=transform)

# Use DistributedSampler to partition the dataset among the nodes
train_sampler = DistributedSampler(train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
val_sampler = DistributedSampler(val_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

# Initialize ResNet-18 model
model = resnet18(weights=None, num_classes=10)  # CIFAR-10 has 10 classes

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Prepare everything with accelerator
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def evaluate_model(model, dataloader, loss_fn, accelerator):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    device = accelerator.device  # Get the device from the accelerator

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the right device
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * len(targets)
            batch_accuracy = compute_accuracy(outputs, targets)
            total_accuracy += batch_accuracy * len(targets)

    return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset)

def train(model, train_loader, optimizer, loss_fn, accelerator, num_epochs=10):
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    device = accelerator.device  # Get the device from the accelerator
    print(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the right device
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            accelerator.backward(loss)
            optimizer.step()

            # Compute accuracy for this batch
            batch_accuracy = compute_accuracy(outputs, targets)
            epoch_accuracy += batch_accuracy * len(targets)
            epoch_loss += loss.item() * len(targets)

        # Compute average loss and accuracy for the epoch
        epoch_loss /= len(train_loader.dataset)
        epoch_accuracy /= len(train_loader.dataset)
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Evaluate the model on the validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fn, accelerator)
        if accelerator.is_local_main_process:
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Run training
train(model, train_loader, optimizer, loss_fn, accelerator)
