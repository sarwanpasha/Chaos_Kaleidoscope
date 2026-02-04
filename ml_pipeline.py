import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from collections import defaultdict

# ============================================================================
# Custom Dataset Class
# ============================================================================

class ProteinImageDataset(Dataset):
    """Dataset class for loading protein kaleidoscope images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels corresponding to images
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Create label to index mapping
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        return image, label_idx


# ============================================================================
# Custom CNN Models (1-Layer, 2-Layer, 3-Layer, 4-Layer)
# ============================================================================

class BlockLayer(nn.Module):
    """Convolutional block with Conv2d -> ReLU -> MaxPool"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super(BlockLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CustomCNN(nn.Module):
    """Custom CNN with variable number of block layers"""
    
    def __init__(self, num_layers, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Build convolutional blocks
        channels = [input_channels, 32, 64, 128, 256]
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.blocks.append(BlockLayer(in_ch, out_ch))
        
        # Calculate feature size after conv blocks
        # Input: 380x380, after each block: size/2
        feature_size = 380
        for _ in range(num_layers):
            feature_size = feature_size // 2
        
        # Final feature dimension
        final_channels = channels[num_layers]
        final_features = final_channels * feature_size * feature_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(final_features, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Pass through convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        
        return x


# ============================================================================
# Pre-trained Models (VGG-19, ResNet-50)
# ============================================================================

def get_pretrained_vgg19(num_classes):
    """Load pre-trained VGG-19 and modify for our number of classes"""
    model = models.vgg19(pretrained=True)
    
    # Modify the classifier for our number of classes
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model


def get_pretrained_resnet50(num_classes):
    """Load pre-trained ResNet-50 and modify for our number of classes"""
    model = models.resnet50(pretrained=True)
    
    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate the model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.5  # Default if calculation fails
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc
    }
    
    return metrics


def train_and_evaluate(model, train_loader, test_loader, num_epochs, learning_rate, device, num_classes):
    """Complete training and evaluation pipeline"""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # NLL loss for multiclass
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to device
    model = model.to(device)
    
    # Track training time
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
    
    # Calculate training time
    train_time = (time.time() - start_time) / 3600  # Convert to hours
    
    # Evaluate on test set
    metrics = evaluate_model(model, test_loader, device, num_classes)
    metrics['train_time'] = train_time
    
    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def load_images_from_directories(base_dir):
    """
    Load image paths and labels from directory structure
    Expected structure:
        base_dir/
            class1/
                0.png
                1.png
                ...
            class2/
                0.png
                1.png
                ...
    """
    image_paths = []
    labels = []
    
    # Get all subdirectories (class labels)
    class_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_dir in class_dirs:
        class_path = os.path.join(base_dir, class_dir)
        
        # Get all image files in this class directory
        image_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)
            labels.append(class_dir)
    
    return image_paths, labels


def run_experiments(image_dir, output_file='results.txt'):
    """
    Run all experiments as described in the paper
    """
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load images and labels
    print("Loading images...")
    image_paths, labels = load_images_from_directories(image_dir)
    print(f"Loaded {len(image_paths)} images from {len(set(labels))} classes")
    
    # Image transformations (as per paper: 380x380 input size)
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Stratified train-test split (80-20)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create datasets
    train_dataset = ProteinImageDataset(train_paths, train_labels, transform=transform)
    test_dataset = ProteinImageDataset(test_paths, test_labels, transform=transform)
    
    num_classes = len(train_dataset.unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.unique_labels}")
    
    # Create data loaders (batch_size=64 as per paper)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Hyperparameters from paper
    learning_rate = 0.003
    num_epochs = 10
    
    # Dictionary to store all results
    all_results = {}
    
    # ========================================================================
    # 1-Layer CNN
    # ========================================================================
    print("\n" + "="*80)
    print("Training 1-Layer CNN...")
    print("="*80)
    model = CustomCNN(num_layers=1, num_classes=num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['1-Layer CNN'] = metrics
    print_metrics('1-Layer CNN', metrics)
    
    # ========================================================================
    # 2-Layer CNN
    # ========================================================================
    print("\n" + "="*80)
    print("Training 2-Layer CNN...")
    print("="*80)
    model = CustomCNN(num_layers=2, num_classes=num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['2-Layer CNN'] = metrics
    print_metrics('2-Layer CNN', metrics)
    
    # ========================================================================
    # 3-Layer CNN
    # ========================================================================
    print("\n" + "="*80)
    print("Training 3-Layer CNN...")
    print("="*80)
    model = CustomCNN(num_layers=3, num_classes=num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['3-Layer CNN'] = metrics
    print_metrics('3-Layer CNN', metrics)
    
    # ========================================================================
    # 4-Layer CNN
    # ========================================================================
    print("\n" + "="*80)
    print("Training 4-Layer CNN...")
    print("="*80)
    model = CustomCNN(num_layers=4, num_classes=num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['4-Layer CNN'] = metrics
    print_metrics('4-Layer CNN', metrics)
    
    # ========================================================================
    # Pre-trained ResNet-50
    # ========================================================================
    print("\n" + "="*80)
    print("Training Pre-trained ResNet-50...")
    print("="*80)
    model = get_pretrained_resnet50(num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['ResNet-50'] = metrics
    print_metrics('ResNet-50', metrics)
    
    # ========================================================================
    # Pre-trained VGG-19
    # ========================================================================
    print("\n" + "="*80)
    print("Training Pre-trained VGG-19...")
    print("="*80)
    model = get_pretrained_vgg19(num_classes)
    metrics = train_and_evaluate(model, train_loader, test_loader, num_epochs, 
                                learning_rate, device, num_classes)
    all_results['VGG-19'] = metrics
    print_metrics('VGG-19', metrics)
    
    # ========================================================================
    # Save all results
    # ========================================================================
    save_results(all_results, output_file)
    
    return all_results


def print_metrics(model_name, metrics):
    """Print metrics in a formatted way"""
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Recall:      {metrics['recall']:.3f}")
    print(f"  F1 (Weighted): {metrics['f1_weighted']:.3f}")
    print(f"  F1 (Macro):   {metrics['f1_macro']:.3f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.3f}")
    print(f"  Train Time:  {metrics['train_time']:.3f} hrs")


def save_results(results, output_file):
    """Save results to a text file in table format"""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DANCE Image Classification Results\n")
        f.write("="*100 + "\n\n")
        
        # Header
        f.write(f"{'Model':<20} {'Acc.':<8} {'Prec.':<8} {'Recall':<8} {'F1(W)':<8} "
                f"{'F1(M)':<8} {'AUC':<8} {'Time(h)':<8}\n")
        f.write("-"*100 + "\n")
        
        # Results for each model
        for model_name, metrics in results.items():
            f.write(f"{model_name:<20} "
                   f"{metrics['accuracy']:<8.3f} "
                   f"{metrics['precision']:<8.3f} "
                   f"{metrics['recall']:<8.3f} "
                   f"{metrics['f1_weighted']:<8.3f} "
                   f"{metrics['f1_macro']:<8.3f} "
                   f"{metrics['roc_auc']:<8.3f} "
                   f"{metrics['train_time']:<8.3f}\n")
        
        f.write("="*100 + "\n")
    
    print(f"\nResults saved to {output_file}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DANCE Image Classification Pipeline')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing class subdirectories with images')
    parser.add_argument('--output', type=str, default='dance_results.txt',
                       help='Output file for results (default: dance_results.txt)')
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_experiments(args.image_dir, args.output)
