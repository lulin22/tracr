import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tenseal as ts
import matplotlib.pyplot as plt
import time
from torch.utils.data import Subset
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Loading CIFAR-10 dataset...")

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Use the full test dataset in the data loader
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Define two CNN variants - one for plain execution and one for HE compatibility
class EnhancedCNN(nn.Module):
    def __init__(self, use_he_friendly=False):
        super(EnhancedCNN, self).__init__()
        self.use_he_friendly = use_he_friendly
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Different pooling for HE vs plain mode
        self.pool = nn.AvgPool2d(2, 2) if use_he_friendly else nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_he_friendly:
            x = x * x  # Square activation (HE-friendly)
        else:
            x = F.relu(x)  # ReLU activation (better accuracy)
        x = self.pool(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_he_friendly:
            x = x * x  # Square activation
        else:
            x = F.relu(x)
        x = self.pool(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_he_friendly:
            x = x * x  # Square activation
        else:
            x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Only use dropout for non-HE mode
        if not self.use_he_friendly:
            x = self.dropout(x)
            
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        if self.use_he_friendly:
            x = x * x  # Square activation
        else:
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x

# Define a compatibility wrapper class to preserve the same interface as before
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)  # Add batch norm
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 120)
        self.bn2 = nn.BatchNorm1d(120)  # Add batch norm
        self.fc2 = nn.Linear(120, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch norm
        x = x * x  # Square activation
        x = self.pool(x)
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        x = self.bn2(x)  # Apply batch norm
        x = x * x  # Square activation
        x = self.fc2(x)
        return x

# Load or train the model
def load_or_train_model(use_enhanced_model=True, use_he_friendly=False):
    if use_enhanced_model:
        model = EnhancedCNN(use_he_friendly=use_he_friendly)
        model_filename = 'cifar_enhanced_cnn.pth'
    else:
        model = ImprovedCNN()
        model_filename = 'cifar_improved_cnn.pth'
    
    try:
        # Try to load pre-trained model
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded pre-trained model from {model_filename}")
    except:
        print(f"Training new model: {'Enhanced CNN' if use_enhanced_model else 'Original CNN'}")
        # Train the model if no pre-trained model exists
        
        # Enhanced data augmentation for better generalization
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Add rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add color jitter
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=train_transform)
        train_size = int(0.9 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Train for more epochs (30 instead of 15)
        num_epochs = 30 if use_enhanced_model else 15
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()  # Set model to training mode
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            
            # Validation check at end of each epoch
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            print(f'Epoch {epoch+1}, Validation accuracy: {100 * val_acc:.2f}%')
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_{model_filename}')
                print(f'New best model saved with accuracy: {100 * best_val_acc:.2f}%')
        
        # Load the best model after training
        try:
            model.load_state_dict(torch.load(f'best_{model_filename}'))
            print('Loaded best model from training')
        except:
            print('Could not load best model, using last epoch model')
        
        print('Finished Training')
        torch.save(model.state_dict(), model_filename)
    
    # Set to evaluation mode
    model.eval()
    return model

# Set up TenSEAL context
def setup_context():
    print("Setting up TenSEAL context...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, 26, 26, 26, 26, 26, 26, 31]
    )
    context.global_scale = 2**26
    context.generate_galois_keys()
    return context

# Function to extract kernel and bias from conv layer
def get_conv_params(model):
    kernels = model.conv1.weight.detach().numpy()
    bias = model.conv1.bias.detach().numpy()
    return kernels, bias

# Function to extract parameters from second conv layer
def get_conv2_params(model):
    kernels = model.conv2.weight.detach().numpy()
    bias = model.conv2.bias.detach().numpy()
    return kernels, bias

# Function to extract parameters from third conv layer
def get_conv3_params(model):
    kernels = model.conv3.weight.detach().numpy()
    bias = model.conv3.bias.detach().numpy()
    return kernels, bias

# Function to extract weights and bias from the first fully connected layer
def get_fc_params(model):
    weights = model.fc1.weight.detach().numpy()
    bias = model.fc1.bias.detach().numpy()
    return weights, bias

# Function to perform im2col for convolution
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    n_channels, height, width = input_data.shape
    
    # Calculate output size
    out_h = (height + 2*pad - filter_h)//stride + 1
    out_w = (width + 2*pad - filter_w)//stride + 1
    
    # Pad input data
    img = np.pad(input_data, [(0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((n_channels, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, y, x, :, :] = img[:, y:y_max:stride, x:x_max:stride]
    
    col = col.reshape(n_channels*filter_h*filter_w, out_h*out_w)
    return col

# Function to perform convolution using im2col and matrix multiplication
def convolve(image, kernels, bias, pad=0):
    n_filters, in_channels, filter_h, filter_w = kernels.shape
    
    # im2col transformation
    col = im2col(image, filter_h, filter_w, stride=1, pad=pad)
    
    # Reshape kernels for matrix multiplication
    kernels_reshaped = kernels.reshape(n_filters, -1)
    
    # Perform convolution by matrix multiplication
    out = np.dot(kernels_reshaped, col) + bias.reshape(-1, 1)
    
    # Calculate output dimensions
    out_h = (image.shape[1] + 2*pad - filter_h) // 1 + 1
    out_w = (image.shape[2] + 2*pad - filter_w) // 1 + 1
    
    # Reshape output to proper dimensions
    out = out.reshape(n_filters, out_h, out_w)
    
    return out

# Define a HE-friendly activation function that performs better than square
def lecun_activation(x):
    return x * torch.sigmoid(x)  # x * sigmoid(x) is polynomial-approximable

# Main function to test CIFAR-10 classification with or without HE
def main():
    # Set to True to use the enhanced model, False to use the original model
    use_enhanced_model = True
    
    # Set to True to use HE-friendly model (square activation, avg pooling)
    use_he_friendly = True
    
    # Force training a new model
    force_new_training = True
    
    # Load or train the model (will train since force_new_training=True)
    model = load_or_train_model(use_enhanced_model=use_enhanced_model, use_he_friendly=use_he_friendly)
    
    # Evaluate the model on plain (non-encrypted) test set
    correct_plain = 0
    total = 0
    class_correct_plain = [0] * 10
    class_total_plain = [0] * 10

    # Add timing for plain model
    plain_start_time = time.time()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_plain += (predicted == labels).sum().item()
            
            # Per-class accuracy
            label = labels[0].item()
            class_total_plain[label] += 1
            if predicted == labels:
                class_correct_plain[label] += 1

    # Calculate plain model timing
    plain_end_time = time.time()
    plain_total_time = plain_end_time - plain_start_time
    plain_avg_time = plain_total_time / total

    print(f'Accuracy of the plain model on {total} test images: {100 * correct_plain / total:.2f}%')
    print(f'Plain model total time: {plain_total_time:.2f} seconds')
    print(f'Plain model average time per image: {plain_avg_time:.4f} seconds')
    
    # Print per-class accuracy for plain model
    print("\nPer-class Accuracy (Plain Model):")
    for i in range(10):
        if class_total_plain[i] > 0:
            accuracy = 100 * class_correct_plain[i] / class_total_plain[i]
            print(f'{classes[i]}: {accuracy:.2f}% ({class_correct_plain[i]}/{class_total_plain[i]})')
    
    # Skip HE testing completely
    print("\n" + "="*50)
    print("Final Results:")
    print(f"Plain model accuracy: {100 * correct_plain / total:.2f}%")
    print("HE encryption evaluation skipped")
    print("="*50)

if __name__ == "__main__":
    main()