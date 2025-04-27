import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import yaml
from network_model import NetworkAnomalyDetector
from network_data_utils import NetworkPacketDataset
from tensorboardX import SummaryWriter
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, phase='train'):
    """Plot confusion matrix and save it"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {phase}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create directory if it doesn't exist
    os.makedirs('confusion_matrices', exist_ok=True)
    plt.savefig(f'confusion_matrices/{phase}_confusion_matrix_final.png')
    plt.close()
    
    # Print classification report
    print(f"\n{phase.capitalize()} Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return cm

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target, _) in enumerate(train_loader):  # Ignore meta
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Store predictions and targets for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        if batch_idx % 10 == 0:  # Print more frequently for smaller dataset
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total, all_preds, all_targets

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target, _ in val_loader:  # Ignore meta
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_targets

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target, _ in test_loader:  # Ignore meta
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Plot final test confusion matrix
    plot_confusion_matrix(all_targets, all_preds, 'test')
    
    return running_loss / len(test_loader), 100. * correct / total

def main():
    # Load configuration
    with open('snippets/model_config_network.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = NetworkAnomalyDetector(config).to(device)
    print("Model created successfully")
    
    # Create dataset
    dataset = NetworkPacketDataset(
        data_path='DATA/ECU_IoHT.xlsx',
        transform=None,
        is_train=True
    )
    
    # Split dataset into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Split train dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders with smaller batch size for ECU_IoHT dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Smaller batch size
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # Smaller batch size
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,  # Smaller batch size
        shuffle=False
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Use weighted cross entropy loss to handle class imbalance
    num_samples = len(train_dataset)
    class_counts = [
        sum(1 for _, label, _ in train_dataset if label == 0),
        sum(1 for _, label, _ in train_dataset if label == 1)
    ]
    weights = torch.FloatTensor([num_samples / (2 * count) for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Create tensorboard writer
    writer = SummaryWriter('logs/training_ecu_ioht')
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    print("Starting training...")
    for epoch in range(config['nb_epochs']):
        print(f"\nEpoch {epoch+1}/{config['nb_epochs']}:")
        
        # Train
        train_loss, train_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best_model_ecu_ioht.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= config['early_stopping']:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Plot final confusion matrices for train and validation
    plot_confusion_matrix(train_targets, train_preds, 'train')
    plot_confusion_matrix(val_targets, val_preds, 'validation')
    
    # Load best model and test
    model.load_state_dict(best_model_state)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    main() 