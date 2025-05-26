import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from network_model import NetworkAnomalyDetector
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CICIoMTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Convert string labels to numerical values
        unique_labels = self.data['label'].unique()
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = self.data['label'].map(self.label_map).values
        self.features = self.data.drop('label', axis=1).values
        
        # Print label information
        print(f"\nDataset: {csv_file}")
        print(f"Number of unique classes: {len(unique_labels)}")
        print("Class mapping:")
        for label, idx in self.label_map.items():
            print(f"  {label}: {idx}")
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Reshape for the model (1, sequence_length)
        feature = torch.FloatTensor(feature).unsqueeze(0)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('Training:')
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print('\nValidation:')
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f'Batch: {batch_idx+1}/{len(val_loader)} | '
                          f'Loss: {loss.item():.4f} | '
                          f'Acc: {100.*val_correct/val_total:.2f}%')
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print epoch summary
        print(f'\nEpoch Summary:')
        print(f'Training Loss: {train_loss:.4f} | Training Acc: {100.*correct/total:.2f}%')
        print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {100.*val_correct/val_total:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('New best model saved!')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load datasets first to get number of classes
    train_dataset = CICIoMTDataset('C:/Users/yildi/OneDrive/Masaüstü/IoMT-Anomaly-Detection-With-Rawnet-2/DATA/CIC_IoMT_2024_WiFi_MQTT_train.csv')
    test_dataset = CICIoMTDataset('C:/Users/yildi/OneDrive/Masaüstü/IoMT-Anomaly-Detection-With-Rawnet-2/DATA/CIC_IoMT_2024_WiFi_MQTT_test.csv')
    
    # Model configuration
    config = {
        'first_conv': 32,
        'filts': [(32, 32), (64, 64), (128, 128)],
        'gru_node': 128,
        'nb_gru_layer': 2,
        'nb_fc_node': 256,
        'nb_classes': len(train_dataset.label_map)  # Set number of classes based on dataset
    }
    
    # Create model
    model = NetworkAnomalyDetector(config).to(device)
    model.summary()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    
    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main() 