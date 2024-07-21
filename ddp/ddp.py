from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18
from accelerate.utils import set_seed

set_seed(0)

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
train_sampler = DistributedSampler(train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)
val_sampler = DistributedSampler(val_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=25, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=25, sampler=val_sampler)

# Initialize ResNet-18 model
model = resnet18(weights=None, num_classes=10)  # CIFAR-10 has 10 classes

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001 * accelerator.num_processes)
loss_fn = nn.CrossEntropyLoss()

# Prepare everything with accelerator
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct

def evaluate_model(model, dataloader, loss_fn, accelerator):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    device = accelerator.device  # Get the device from the accelerator

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the right device
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * len(targets)
            total_correct += compute_accuracy(outputs, targets)
            total_samples += len(targets)

    # Gather results from all processes
    total_loss = torch.tensor(total_loss, device=device)
    total_correct = torch.tensor(total_correct, device=device)
    total_samples = torch.tensor(total_samples, device=device)

    total_loss = accelerator.gather(total_loss).sum().item()
    total_correct = accelerator.gather(total_correct).sum().item()
    total_samples = accelerator.gather(total_samples).sum().item()

    return total_loss / total_samples, total_correct / total_samples

def train(model, train_loader, optimizer, loss_fn, accelerator, num_epochs=10):
    device = accelerator.device  # Get the device from the accelerator
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the right device
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            accelerator.backward(loss)
            optimizer.step()

            # Compute accuracy for this batch
            total_correct += compute_accuracy(outputs, targets)
            epoch_loss += loss.item() * len(targets)
            total_samples += len(targets)

        # Gather results from all processes
        epoch_loss = torch.tensor(epoch_loss, device=device)
        total_correct = torch.tensor(total_correct, device=device)
        total_samples = torch.tensor(total_samples, device=device)

        epoch_loss = accelerator.gather(epoch_loss).sum().item()
        total_correct = accelerator.gather(total_correct).sum().item()
        total_samples = accelerator.gather(total_samples).sum().item()

        epoch_loss /= total_samples
        epoch_accuracy = total_correct / total_samples
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Evaluate the model on the validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fn, accelerator)
        if accelerator.is_local_main_process:
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Run training
train(model, train_loader, optimizer, loss_fn, accelerator)
