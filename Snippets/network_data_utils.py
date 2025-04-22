import torch
import collections
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define a named tuple for network data
NetworkData = collections.namedtuple('NetworkData',
    ['id', 'features', 'label'])

class NetworkPacketDataset(Dataset):
    """Dataset class for network traffic features"""
    def __init__(self, data_path=None, transform=None, 
                 is_train=True, sample_size=None):
        
        self.data_path = data_path
        self.transform = transform
        self.is_train = is_train
        
        # Load the data
        print(f"Loading data from {self.data_path}...")
        file_ext = os.path.splitext(self.data_path)[1].lower()
        if file_ext == '.xlsx' or file_ext == '.xls':
            self.df = pd.read_excel(self.data_path)
        elif file_ext == '.csv':
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Print dataset information
        print("\nDataset Info:")
        print("Columns:", self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nLabel (Type) distribution:")
        print(self.df['Type'].value_counts())
        
        # Convert categorical features to numeric
        numeric_features = ['No.', 'Time', 'Length']
        categorical_features = ['Source', 'Destination', 'Protocol', 'Info']
        
        # Initialize label encoder for categorical features
        self.label_encoders = {}
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.df[feature] = self.label_encoders[feature].fit_transform(self.df[feature])
        
        # Combine features
        self.feature_columns = numeric_features + categorical_features
        
        # Convert features to float32
        self.features = self.df[self.feature_columns].astype('float32').values
        
        # Convert labels (Type column) to binary (0 for Normal, 1 for Attack)
        self.labels = (self.df['Type'] == 'Attack').astype('int64').values
        
        # Normalize features
        if is_train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            # Use the scaler from the training set
            if not hasattr(NetworkPacketDataset, 'train_scaler'):
                raise ValueError("No training scaler found. Train dataset must be created before validation dataset.")
            self.features = NetworkPacketDataset.train_scaler.transform(self.features)
        
        # Store the scaler for validation set
        if is_train:
            NetworkPacketDataset.train_scaler = self.scaler
        
        # Create metadata
        self.data_meta = [
            NetworkData(
                id=idx,
                features=features,
                label=label
            )
            for idx, (features, label) in enumerate(zip(self.features, self.labels))
        ]
        
        # Sample a subset if requested
        if sample_size:
            indices = np.random.choice(len(self.data_meta), 
                                     size=(sample_size,), 
                                     replace=False)
            self.data_meta = [self.data_meta[i] for i in indices]
            self.features = self.features[indices]
            self.labels = self.labels[indices]
        
        self.length = len(self.features)
        print(f"\nLoaded {self.length} samples")
        print(f"Number of features: {self.features.shape[1]}")
        print(f"Label distribution: {collections.Counter(self.labels)}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        # Reshape the features to [channels, sequence_length]
        x = x.unsqueeze(0)  # Add channel dimension [1, sequence_length]
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y, self.data_meta[idx]
    
    def _process_data(self):
        """Process the data into a format suitable for the model"""
        files_meta = []
        self.data_x = []
        self.data_y = []
        
        required_columns = ['packet_data', 'label']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found. Available columns: {self.df.columns.tolist()}")
        
        print("Processing data...")
        for idx, row in self.df.iterrows():
            try:
                # Extract packet data
                packet_data = row['packet_data']
                
                # Convert packet data to a list of integers
                if isinstance(packet_data, str):
                    # Handle different string formats
                    packet_data = packet_data.strip('[]() ').replace(' ', '')
                    if ',' in packet_data:
                        packet_bytes = [float(b) for b in packet_data.split(',') if b]
                    else:
                        packet_bytes = [float(b) for b in packet_data.split() if b]
                elif isinstance(packet_data, (list, np.ndarray)):
                    packet_bytes = list(packet_data)
                else:
                    raise ValueError(f"Unsupported packet data type: {type(packet_data)}")
                
                # Pad or truncate to max_len
                if len(packet_bytes) > self.max_len:
                    packet_bytes = packet_bytes[:self.max_len]
                else:
                    packet_bytes = packet_bytes + [0] * (self.max_len - len(packet_bytes))
                
                # Normalize if requested
                if self.normalize:
                    packet_bytes = [(b - 128) / 128.0 for b in packet_bytes]
                
                # Get label
                label = int(row['label'])
                
                # Create metadata
                meta = NetworkData(
                    id=idx,
                    features=packet_bytes,
                    label=label
                )
                
                files_meta.append(meta)
                self.data_x.append(packet_bytes)
                self.data_y.append(label)
                
                if idx % 1000 == 0:
                    print(f"Processed {idx} samples...")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(files_meta)} samples")
        print(f"Label distribution: {collections.Counter(self.data_y)}")
        return files_meta 