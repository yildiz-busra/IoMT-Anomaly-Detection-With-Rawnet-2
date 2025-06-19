# Network Packet Anomaly Detection using RawNet

This project adapts the RawNet model, originally designed for audio spoofing detection, to detect anomalies in network packets. The model uses raw packet data as input and learns to distinguish between normal and anomalous network traffic.

## Project Structure

- `network_data_utils.py`: Contains the `NetworkPacketDataset` class for loading and processing network packet data from Excel files.
- `network_main.py`: Main script for training and evaluating the model.
- `model_config_network.yaml`: Configuration file for the RawNet model adapted for network packets.
- `prepare_data.py`: Utility script to prepare Excel data for training.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- pandas
- numpy
- tensorboardX
- pyyaml

## Data Preparation

Before training the model, you need to prepare your network packet data in an Excel file. The Excel file should contain at least two columns:
- A column with packet data (raw bytes)
- A column with labels (0 for normal, 1 for anomaly)

Use the `prepare_data.py` script to prepare your data:

```bash
python prepare_data.py --input DATA/your_data.xlsx --output DATA/prepared_data.xlsx --label_column label --data_column packet_data
```

## Training the Model

To train the model, use the `network_main.py` script:

```bash
python network_main.py --data_path DATA/prepared_data.xlsx --num_epochs 100 --batch_size 32 --lr 0.0001 --max_len 1024
```

### Command-line Arguments

- `--data_path`: Path to the Excel file containing network packet data (default: 'DATA/ECU_IoHT.xlsx')
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--weight_decay`: Weight decay for optimizer (default: 0.0001)
- `--max_len`: Maximum length of packet data (default: 1024)
- `--config`: Path to model configuration file (default: 'model_config_RawNet2.yaml')
- `--comment`: Comment to describe the saved model
- `--eval`: Run in evaluation mode
- `--model_path`: Path to a pre-trained model checkpoint
- `--eval_output`: Path to save evaluation results

## Evaluating the Model

To evaluate a trained model:

```bash
python network_main.py --eval --model_path models/your_model.pth --data_path DATA/prepared_data.xlsx --eval_output results.txt
```

## Model Architecture

The model uses the RawNet architecture, which consists of:
1. SincConv layer for initial feature extraction
2. Residual blocks for deep feature learning
3. Attention mechanism to focus on important parts of the packet
4. GRU layers for sequential modeling
5. Fully connected layers for classification

## Adapting to Your Data

If your network packet data has a different format or structure, you may need to modify the `NetworkPacketDataset` class in `network_data_utils.py` to properly load and process your data.

## References

- RawNet: [https://github.com/Jungjee/RawNet](https://github.com/Jungjee/RawNet)
- ASVspoof 2019: [https://www.asvspoof.org/](https://www.asvspoof.org/) 