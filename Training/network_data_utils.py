import torch
import collections
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Define a named tuple for network data
NetworkData = collections.namedtuple('NetworkData',
    ['id', 'features', 'label'])

class NetworkPacketDataset(Dataset):
    
    def __init__(self, data_path=None, transform=None, 
                 is_train=True, sample_size=None, class_config=19):
        
        self.data_path = data_path
        self.transform = transform
        self.is_train = is_train
        self.class_config = class_config
        
        # Load the data
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Print dataset information
        print("\nDataset Info:")
        print("Columns:", self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nLabel distribution:")
        print(self.df['label'].value_counts())
        
        # Process the data
        self._process_data()
        
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=42)
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.df.drop(['label'], axis=1).values)
        self.labels = torch.LongTensor(self.df['label'].values)
        
        print(f"\nFinal dataset shape: {self.features.shape}")
        print(f"Number of classes: {len(np.unique(self.labels))}")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label
    
    def _process_data(self):
        
        print("\nProcessing data...")
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['timestamp', 'device_id', 'device_type', 'device_name']
        self.df = self.df.drop([col for col in columns_to_drop if col in self.df.columns], axis=1)
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
        # Convert categorical features to numeric
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'label']
        
        # Initialize label encoder for categorical features
        self.label_encoders = {}
        for feature in categorical_columns:
            self.label_encoders[feature] = LabelEncoder()
            self.df[feature] = self.label_encoders[feature].fit_transform(self.df[feature])
        
        # Clean up labels by removing train/test suffixes
        self.df['label'] = self.df['label'].str.replace('_train', '').str.replace('_test', '')
        
        # Map attack categories based on class_config
        if self.class_config == 2:
            self.df['label'] = self.df['label'].apply(lambda x: 1 if x != 'Benign' else 0)
        elif self.class_config == 6:
            # Map to 6 categories
            category_mapping = {
                'ARP_Spoofing': 'Spoofing',
                'MQTT-DDoS-Connect_Flood': 'MQTT',
                'MQTT-DDoS-Publish_Flood': 'MQTT',
                'MQTT-DoS-Connect_Flood': 'MQTT',
                'MQTT-DoS-Publish_Flood': 'MQTT',
                'MQTT-Malformed_Data': 'MQTT',
                'Recon-OS_Scan': 'Recon',
                'Recon-Ping_Sweep': 'Recon',
                'Recon-Port_Scan': 'Recon',
                'Recon-VulScan': 'Recon',
                'TCP_IP-DDoS-ICMP': 'DDoS',
                'TCP_IP-DDoS-SYN': 'DDoS',
                'TCP_IP-DDoS-TCP': 'DDoS',
                'TCP_IP-DDoS-UDP': 'DDoS',
                'TCP_IP-DoS-ICMP': 'DoS',
                'TCP_IP-DoS-SYN': 'DoS',
                'TCP_IP-DoS-TCP': 'DoS',
                'TCP_IP-DoS-UDP': 'DoS',
                'Benign': 'Benign'
            }
            self.df['label'] = self.df['label'].map(category_mapping)
        
        # Convert labels to numeric
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['label'])
        
        # Print label mapping for verification
        print("\nLabel mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{i}: {label}")
        
        # Normalize numerical features
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = [col for col in numerical_columns if col != 'label']
        
        self.scaler = StandardScaler()
        self.df[numerical_columns] = self.scaler.fit_transform(self.df[numerical_columns])
        
        print("Data processing completed.")
        print(f"Number of features: {len(self.df.columns) - 1}")  # -1 for label column
        print(f"Feature names: {self.df.columns.tolist()[:-1]}")  # Exclude label column 