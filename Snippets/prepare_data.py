import pandas as pd
import numpy as np
import os
import argparse

def prepare_data(input_file, output_file, label_column='label', data_column='packet_data'):
    """
    Prepare data for training the network packet anomaly detection model.
    
    Args:
        input_file: Path to the input file (Excel or CSV)
        output_file: Path to save the prepared file
        label_column: Name of the column containing labels (default: 'label')
        data_column: Name of the column containing packet data (default: 'packet_data')
    """
    print(f"Reading data from {input_file}...")
    
    # Determine file type and read accordingly
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(input_file)
    elif file_ext == '.csv':
        df = pd.read_csv(input_file)
    else:
        print(f"Error: Unsupported file format '{file_ext}'. Please use Excel (.xlsx/.xls) or CSV (.csv) files.")
        return False
    
    # Check if required columns exist
    if label_column not in df.columns:
        print(f"Error: Column '{label_column}' not found in the file.")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    if data_column not in df.columns:
        print(f"Error: Column '{data_column}' not found in the file.")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    # Ensure labels are binary (0 for normal, 1 for anomaly)
    if df[label_column].nunique() > 2:
        print(f"Warning: Label column '{label_column}' contains more than 2 unique values.")
        print(f"Unique values: {df[label_column].unique()}")
        print("Converting to binary labels (0 for normal, 1 for anomaly)...")
        
        # Assuming the most common value is normal (0) and others are anomalies (1)
        most_common = df[label_column].mode()[0]
        df[label_column] = (df[label_column] != most_common).astype(int)
    
    # Ensure packet data is in the correct format
    # If packet data is stored as strings, convert to lists of integers
    if isinstance(df[data_column].iloc[0], str):
        print("Converting string packet data to lists of integers...")
        df[data_column] = df[data_column].apply(lambda x: 
            [int(b) for b in x.strip('[]').split(',')] if isinstance(x, str) else x)
    
    # Save the prepared data
    print(f"Saving prepared data to {output_file}...")
    output_ext = os.path.splitext(output_file)[1].lower()
    if output_ext == '.xlsx' or output_ext == '.xls':
        df.to_excel(output_file, index=False)
    elif output_ext == '.csv':
        df.to_csv(output_file, index=False)
    else:
        print(f"Error: Unsupported output format '{output_ext}'. Please use Excel (.xlsx/.xls) or CSV (.csv) files.")
        return False
    
    print("Data preparation completed successfully!")
    
    # Print some statistics
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {len(df[df[label_column] == 0])}")
    print(f"Anomaly samples: {len(df[df[label_column] == 1])}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for network packet anomaly detection')
    parser.add_argument('--input', type=str, required=True, help='Path to the input file (Excel or CSV)')
    parser.add_argument('--output', type=str, required=True, help='Path to save the prepared file')
    parser.add_argument('--label_column', type=str, default='label', help='Name of the column containing labels')
    parser.add_argument('--data_column', type=str, default='packet_data', help='Name of the column containing packet data')
    
    args = parser.parse_args()
    
    prepare_data(
        args.input, 
        args.output, 
        args.label_column, 
        args.data_column
    ) 