import torch
import torch.nn as nn
import numpy as np

class PacketConv(nn.Module):
    """Convolutional layer for processing network packets"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(PacketConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PacketBlock(nn.Module):
    """Residual block for network packet processing"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PacketBlock, self).__init__()
        self.conv1 = PacketConv(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = PacketConv(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out

class NetworkAnomalyDetector(nn.Module):
    """Network packet anomaly detection model based on RawNet architecture"""
    def __init__(self, config):
        super(NetworkAnomalyDetector, self).__init__()
        
        # Initial convolution
        self.first_conv = PacketConv(1, config['first_conv'], kernel_size=3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        in_channels = config['first_conv']
        for filt in config['filts']:
            self.blocks.append(PacketBlock(in_channels, filt[0]))
            in_channels = filt[0]
            
        # GRU layers for sequence processing
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=config['gru_node'],
            num_layers=config['nb_gru_layer'],
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config['gru_node'] * 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(config['gru_node'] * 2, config['nb_fc_node']),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(config['nb_fc_node'], config['nb_classes'])
        )
        
    def forward(self, x):
        # Initial convolution
        x = self.first_conv(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
            
        # Prepare for GRU (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # GRU processing
        gru_out, _ = self.gru(x)
        
        # Attention mechanism
        attention_weights = self.attention(gru_out)
        attended = torch.sum(attention_weights * gru_out, dim=1)
        
        # Classification
        out = self.fc(attended)
        return out
    
    def summary(self):
        """Print model summary"""
        total_params = 0
        print("\nModel Summary:")
        print("-" * 50)
        
        # Count parameters for each layer
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.GRU)):
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                print(f"{name}: {params:,} parameters")
                
        print("-" * 50)
        print(f"Total parameters: {total_params:,}")
        print("-" * 50) 