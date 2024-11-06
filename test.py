import pandas as pd

# Load the CSV file
df = pd.read_csv('network_flows.csv')

# Display basic information
print(df.head())

# Extract unique devices
devices = pd.unique(df[['source_ip', 'destination_ip']].values.ravel('K'))
device_mapping = {device: idx for idx, device in enumerate(devices)}

# Map IP addresses to node indices
df['source'] = df['source_ip'].map(device_mapping)
df['destination'] = df['destination_ip'].map(device_mapping)

# Convert timestamp to datetime if necessary
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort flows by timestamp
df = df.sort_values('timestamp')

from torch_geometric_temporal.signal import TemporalData

# Define time windows (e.g., 1 second intervals)
df['time_window'] = df['timestamp'].astype(int) // 1e9  # Convert to seconds

# Aggregate flows per time window
aggregated_flows = df.groupby(['time_window', 'source', 'destination']).agg({
    'feature1': 'mean',
    'feature2': 'sum',
    # ... include all relevant features
    'label': 'max'  # Assuming '1' indicates anomaly
}).reset_index()

# Create a list of TemporalData objects
temporal_data = []
time_steps = aggregated_flows['time_window'].unique()

for t in time_steps:
    current_flows = aggregated_flows[aggregated_flows['time_window'] == t]
    edge_index = current_flows[['source', 'destination']].values.T
    edge_attr = current_flows[['feature1', 'feature2']].values  # Include all relevant features
    y = current_flows['label'].values  # Labels for edges

    temporal_data.append(TemporalData(edge_index=edge_index, edge_attr=edge_attr, y=y))

import torch
import torch.nn as nn
from torch_geometric_temporal.nn import TGNMemory, TGNN
from torch_geometric_temporal.dataset import TemporalLinkPredictionDataset
from torch_geometric_temporal.signal import temporal_signal_split

# Define the memory module
memory = TGNMemory(
    node_features=feature_size,  # Number of node features
    memory_dimension=32,
    time_embedding_dimension=32,
    message_dimension=32,
    aggregator='mean'  # Aggregation method for messages
)

# Define the TGN-based model
class TGNLinkPredictor(nn.Module):
    def __init__(self, memory):
        super(TGNLinkPredictor, self).__init__()
        self.tgn = TGNN(
            memory=memory,
            message_dim=32,
            mlp_hidden_dims=[32, 32],
            aggregator="mean"
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr, y, memory):
        h = self.tgn(x, edge_index, edge_attr, memory)
        # Combine node representations (source and destination)
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        h_combined = torch.cat([h_src, h_dst], dim=1)
        out = self.classifier(h_combined)
        return out

model = TGNLinkPredictor(memory)

# Assuming 'temporal_data' is a list of TemporalData objects
train_data, test_data = temporal_signal_split(temporal_data, train_ratio=0.8)

# Example of iterating through the training data
for time, snapshot in enumerate(train_data):
    # Prepare node features (if available)
    node_features = torch.zeros((num_nodes, feature_size))  # Initialize or load actual features

    # Forward pass
    out = model(node_features, snapshot.edge_index, snapshot.edge_attr, snapshot.y, memory)

    # Compute loss (e.g., Binary Cross-Entropy for anomaly detection)
    loss = loss_fn(out, snapshot.y.float())

    # Backpropagation and optimization steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import torch.optim as optim

# Define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for snapshot in train_data:
        # Forward pass
        out = model(node_features, snapshot.edge_index, snapshot.edge_attr, snapshot.y, memory)
        
        # Compute loss
        loss = loss_fn(out.squeeze(), snapshot.y.float())
        total_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_data)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for snapshot in test_data:
        out = model(node_features, snapshot.edge_index, snapshot.edge_attr, snapshot.y, memory)
        preds = (out.squeeze() > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(snapshot.y.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

'''
def predict_anomaly(new_flow, model, memory):
    # Preprocess new_flow similarly to training data
    source = device_mapping[new_flow['source_ip']]
    destination = device_mapping[new_flow['destination_ip']]
    edge_index = torch.tensor([[source], [destination]], dtype=torch.long)
    edge_attr = torch.tensor([new_flow[['feature1', 'feature2']].values], dtype=torch.float)

    # Forward pass
    with torch.no_grad():
        out = model(node_features, edge_index, edge_attr, None, memory)
        anomaly_score = out.item()
        is_anomaly = anomaly_score > 0.5
    return is_anomaly, anomaly_score

# Example usage
new_flow = {
    'source_ip': '198.18.134.99',
    'destination_ip': '198.18.134.2',
    'feature1': 0.5,
    'feature2': 1.2,
    # ... other features
}
is_anomaly, score = predict_anomaly(new_flow, model, memory)
print(f'Anomaly: {is_anomaly}, Score: {score}')

'''