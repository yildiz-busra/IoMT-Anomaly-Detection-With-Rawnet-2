import torch
import torch.nn as nn
import pandas as pd
import yaml
from network_model import NetworkAnomalyDetector
from network_data_utils import NetworkPacketDataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Attack category mappings
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def plot_confusion_matrix(y_true, y_pred, phase='train', class_config=19):
    """Plot confusion matrix and save it"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {phase} (Classes: {class_config})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create directory if it doesn't exist
    os.makedirs('confusion_matrices', exist_ok=True)
    plt.savefig(f'confusion_matrices/{phase}_confusion_matrix_ciciomt_{class_config}classes.png')
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
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return running_loss/len(train_loader), 100.*correct/total, all_preds, all_targets

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
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
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return running_loss/len(val_loader), 100.*correct/total, all_preds, all_targets

def main():
    # Load configuration
    with open('model_config_network.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set class configuration (2, 6, or 19 classes)
    class_config = 6  # Changed from 19 to 6 for 6-class classification
    
    # Load and process training data
    print("\nLoading training data...")
    train_dataset = NetworkPacketDataset(
        data_path='../DATA/CIC_IoMT_2024_WiFi_MQTT_train.csv',
        is_train=True,
        class_config=class_config
    )
    
    # Load and process test data
    print("\nLoading test data...")
    test_dataset = NetworkPacketDataset(
        data_path='../DATA/CIC_IoMT_2024_WiFi_MQTT_test.csv',
        is_train=False,
        class_config=class_config
    )
    
    # Split training data into train and validation (90-10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Create train and validation datasets
    train_data = train_dataset.features[:train_size]
    train_labels = train_dataset.labels[:train_size]
    val_data = train_dataset.features[train_size:]
    val_labels = train_dataset.labels[train_size:]
    test_data = test_dataset.features
    test_labels = test_dataset.labels
    
    # Create DataLoader objects
    batch_size = 4096  # Optimized for 16GB RAM with 10GB for training
    num_workers = 3  # Using 3 workers (6 logical cores) + 1 core for main process
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Number of batches loaded in advance by each worker
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Update number of classes in config based on class_config
    config['nb_classes'] = class_config
    
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
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Plot confusion matrices
        plot_confusion_matrix(train_targets, train_preds, 'train', class_config)
        plot_confusion_matrix(val_targets, val_preds, 'validation', class_config)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/best_model_ciciomt_{class_config}classes.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config['early_stopping']:
                print("Early stopping triggered")
                break
    
    # Test best model
    model.load_state_dict(torch.load(f'models/best_model_ciciomt_{class_config}classes.pth'))
    test_loss, test_acc, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )
    test_losses.append(test_loss)
    print(f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    plot_confusion_matrix(test_targets, test_preds, 'test', class_config)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves (Classes: {class_config})')
    plt.legend()
    plt.savefig(f'loss_curves_ciciomt_{class_config}classes.png')
    plt.close()

if __name__ == '__main__':
    main() 