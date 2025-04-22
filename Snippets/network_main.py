import argparse
import sys
import os
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
from network_data_utils import NetworkPacketDataset
from model import RawNet

def pad(x, max_len=1024):
    """Pad or truncate packet data to a fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def init_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:
            print('no weight', m)

def evaluate_accuracy(data_loader, model, device):
    """Evaluate model accuracy on a dataset"""
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path):
    """Generate evaluation results file"""
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y, is_test=True)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend([meta.file_name for meta in batch_meta])
        key_list.extend(['normal' if key == 1 else 'anomaly' for key in list(batch_y.cpu().numpy())])
        score_list.extend(batch_score.tolist())
        
    with open(save_path, 'w') as fh:
        for f, k, cm in zip(fname_list, key_list, score_list):
            fh.write('{} {} {}\n'.format(f, k, cm))
    print('Result saved to {}'.format(save_path))

def train_epoch(data_loader, model, lr, optim, device):
    """Train the model for one epoch"""
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    
    # Adjust class weights based on your dataset's class distribution
    # For anomaly detection, you might want to give more weight to anomaly samples
    weight = torch.FloatTensor([1.0, 9.0]).to(device)  # Adjust weights as needed
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format((num_correct/num_total)*100))
            
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Network Packet Anomaly Detection')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--data_path', type=str, default='DATA/ECU_IoHT.xlsx', 
                        help='Path to the Excel file containing network packet data')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--max_len', type=int, default=1024,
                        help='Maximum length of packet data')
    parser.add_argument('--config', type=str, default='model_config_RawNet2.yaml',
                        help='Path to model configuration file')
    
    args = parser.parse_args()
    
    # Load model configuration
    with open(args.config, 'r') as f_yaml:
        config = yaml.load(f_yaml)
    
    # Set random seeds for reproducibility
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')
    
    # Create model tag
    model_tag = 'model_network_{}_{}_{}_{}'.format(
        args.num_epochs, args.batch_size, args.lr, args.max_len)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # Define transformations
    transforms_list = [
        lambda x: pad(x, max_len=args.max_len),
        lambda x: Tensor(x)
    ]
    transform = transforms.Compose(transforms_list)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = NetworkPacketDataset(
        data_path=args.data_path,
        transform=transform,
        is_train=True,
        max_len=args.max_len,
        normalize=True
    )
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = RawNet(config['model'], device).to(device)
    model.apply(init_weights)
    
    # Load pre-trained model if specified
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded: {}'.format(args.model_path))
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Evaluation mode
    if args.eval:
        produce_evaluation_file(val_dataset, model, device, args.eval_output)
        sys.exit(0)
    
    # Training loop
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 0
    for epoch in range(args.num_epochs):
        running_loss, train_accuracy = train_epoch(
            train_loader, model, args.lr, optimizer, device)
        valid_accuracy = evaluate_accuracy(val_loader, model, device)
        
        # Log metrics
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        
        print('\n{} - {} - {:.2f} - {:.2f}'.format(
            epoch, running_loss, train_accuracy, valid_accuracy))
        print('*'*50)
        print('val_acc %f', valid_accuracy)
        
        # Save best model
        if valid_accuracy > best_acc:
            print('best model found at epoch', epoch)
            best_acc = valid_accuracy
            torch.save(model.state_dict(), 
                      os.path.join(model_save_path, 'best_model.pth'))
        
        # Save checkpoint
        torch.save(model.state_dict(), 
                  os.path.join(model_save_path, f'epoch_{epoch}.pth'))
        
        print('*'*50) 