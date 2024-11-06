# preprocess.py

import os
import zipfile
import pandas as pd
from pathlib import Path
from torch_geometric_temporal.signal import TemporalData
import numpy as np

# Define paths
DATA_DIR = Path('data/reduced/flows')  # Use 'extended' for the extended dataset
OUTPUT_DIR = Path('preprocessed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define feature columns (adjust based on actual CSV structure)
FEATURE_COLUMNS = [
    'feature1', 'feature2',  # Replace with actual feature names
    # Add all relevant feature columns here
]

LABEL_COLUMN = 'label'

def unzip_files(input_dir, output_dir):
    for zip_file in input_dir.glob('*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f'Unzipped {zip_file} to {output_dir}')

def load_and_merge_flows(flows_dir):
    all_dfs = []
    for zip_file in flows_dir.glob('*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.csv'):
                    with zip_ref.open(file) as f:
                        df = pd.read_csv(f)
                        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f'Merged {len(all_dfs)} CSV files into a single DataFrame with {len(merged_df)} rows')
    return merged_df

def preprocess_data(df):
    # Extract unique devices
    devices = pd.unique(df[['source_ip', 'destination_ip']].values.ravel('K'))
    device_mapping = {device: idx for idx, device in enumerate(devices)}
    num_nodes = len(device_mapping)
    print(f'Found {num_nodes} unique devices')

    # Map IP addresses to node indices
    df['source'] = df['source_ip'].map(device_mapping)
    df['destination'] = df['destination_ip'].map(device_mapping)

    # Handle missing mappings if any
    df.dropna(subset=['source', 'destination'], inplace=True)
    df['source'] = df['source'].astype(int)
    df['destination'] = df['destination'].astype(int)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort flows by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Define time windows (e.g., 1 second intervals)
    # Convert timestamp to UNIX epoch in seconds
    df['time_window'] = df['timestamp'].astype(np.int64) // 10**9

    # Aggregate flows per time window
    aggregated_flows = df.groupby(['time_window', 'source', 'destination']).agg({
        **{feature: 'mean' for feature in FEATURE_COLUMNS},
        LABEL_COLUMN: 'max'  # Assuming '1' indicates anomaly
    }).reset_index()

    print(f'Aggregated flows into {aggregated_flows["time_window"].nunique()} time windows')

    return aggregated_flows, device_mapping, num_nodes

def create_temporal_data(aggregated_flows, num_nodes):
    temporal_data = []
    time_steps = aggregated_flows['time_window'].unique()
    print(f'Creating TemporalData objects for {len(time_steps)} time steps')

    for t in time_steps:
        current_flows = aggregated_flows[aggregated_flows['time_window'] == t]
        if current_flows.empty:
            continue

        edge_index = torch.tensor(current_flows[['source', 'destination']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(current_flows[FEATURE_COLUMNS].values, dtype=torch.float)
        y = torch.tensor(current_flows[LABEL_COLUMN].values, dtype=torch.float)

        temporal_data.append(TemporalData(edge_index=edge_index, edge_attr=edge_attr, y=y))

    return temporal_data

def save_preprocessed_data(temporal_data, device_mapping, num_nodes):
    torch.save({
        'temporal_data': temporal_data,
        'device_mapping': device_mapping,
        'num_nodes': num_nodes
    }, OUTPUT_DIR / 'preprocessed_data.pt')
    print(f'Saved preprocessed data to {OUTPUT_DIR / "preprocessed_data.pt"}')

def main():
    # Step 1: Unzip all flow files
    unzip_files(DATA_DIR, OUTPUT_DIR)

    # Step 2: Load and merge all flows
    merged_df = load_and_merge_flows(OUTPUT_DIR)

    # Step 3: Preprocess data
    aggregated_flows, device_mapping, num_nodes = preprocess_data(merged_df)

    # Step 4: Create TemporalData objects
    temporal_data = create_temporal_data(aggregated_flows, num_nodes)

    # Step 5: Save preprocessed data
    save_preprocessed_data(temporal_data, device_mapping, num_nodes)

if __name__ == '__main__':
    import torch
    main()
