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
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.unsqueeze(1)  # Add channel dimension
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(1)  # Add channel dimension
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets

def main():
    # Load configuration
    with open('model_config_network.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = NetworkPacketDataset(
        data_path='..\DATA\wustl-ehms-2020_with_attacks_categories.csv',
        is_train=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Create model
    model = NetworkAnomalyDetector(config).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_acc = 0
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    test_losses = []
    
    for epoch in range(config['nb_epochs']):
        print(f"\nEpoch {epoch+1}/{config['nb_epochs']}")
        
        # Train
        train_loss, train_acc, train_preds, train_targets = train(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(
            model, test_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Plot confusion matrices
        plot_confusion_matrix(train_targets, train_preds, 'train')
        plot_confusion_matrix(val_targets, val_preds, 'validation')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_model_wustl.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config['early_stopping']:
                print("Early stopping triggered")
                break
    
    # Test best model
    model.load_state_dict(torch.load('models/best_model_wustl.pth'))
    test_loss, test_acc, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )
    test_losses.append(test_loss)
    print(f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    plot_confusion_matrix(test_targets, test_preds, 'test')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

if __name__ == '__main__':
    main() 