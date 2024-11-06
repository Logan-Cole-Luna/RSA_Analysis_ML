# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn import TGNMemory, TGNN
from torch_geometric_temporal.signal import TemporalData
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Define paths
PREPROCESSED_DATA_PATH = Path('preprocessed/preprocessed_data.pt')
MODEL_SAVE_PATH = Path('models/tgn_model.pt')

# Load preprocessed data
def load_preprocessed_data(path):
    data = torch.load(path)
    temporal_data = data['temporal_data']
    device_mapping = data['device_mapping']
    num_nodes = data['num_nodes']
    return temporal_data, device_mapping, num_nodes

# Define feature size based on FEATURE_COLUMNS in preprocessing
FEATURE_SIZE = 50  # Update based on actual number of features

# Define the TGN model
class TGNLinkPredictor(nn.Module):
    def __init__(self, memory):
        super(TGNLinkPredictor, self).__init__()
        self.tgn = TGNN(
            memory=memory,
            message_dim=32,
            mlp_hidden_dims=[32, 32],
            aggregator="mean"
        )
        # Assuming node embeddings are of size 32
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),  # Concatenated source and destination embeddings
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

def main():
    # Step 1: Load data
    temporal_data, device_mapping, num_nodes = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    print(f'Loaded preprocessed data with {len(temporal_data)} time steps and {num_nodes} nodes')

    # Step 2: Initialize TGN Memory
    memory = TGNMemory(
        node_features=FEATURE_SIZE,
        memory_dimension=32,
        time_embedding_dimension=32,
        message_dimension=32,
        aggregator='mean'
    )

    # Step 3: Initialize the model
    model = TGNLinkPredictor(memory)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 4: Split data into training and testing
    train_data, test_data = temporal_signal_split(temporal_data, train_ratio=0.8)
    print(f'Training on {len(train_data)} time steps, Testing on {len(test_data)} time steps')

    # Step 5: Define loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 6: Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for snapshot in train_data:
            # Prepare node features (if available)
            # Here we assume node features are not dynamic and set to zeros or load actual features
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            node_features = torch.zeros((num_nodes, FEATURE_SIZE)).to(device)  # Replace with actual features if available

            # Move snapshot to device
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)

            # Forward pass
            out = model(node_features, edge_index, edge_attr, y, memory)

            # Compute loss
            loss = loss_fn(out.squeeze(), y.float())
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Step 7: Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    # Step 8: Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for snapshot in test_data:
            # Prepare node features
            node_features = torch.zeros((num_nodes, FEATURE_SIZE)).to(device)  # Replace with actual features if available

            # Move snapshot to device
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)

            # Forward pass
            out = model(node_features, edge_index, edge_attr, y, memory)
            preds = (out.squeeze() > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f'Evaluation Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Optional: Save evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    torch.save(metrics, OUTPUT_DIR / 'evaluation_metrics.pt')
    print('Saved evaluation metrics')

    # Step 9: Inference Function (Optional)
    def predict_anomaly(new_flow, model, memory, device='cpu'):
        """
        Predict whether a new flow is anomalous.
        new_flow: dict containing 'source_ip', 'destination_ip', and feature values.
        """
        # Preprocess new_flow
        source_ip = new_flow['source_ip']
        destination_ip = new_flow['destination_ip']

        if source_ip not in device_mapping or destination_ip not in device_mapping:
            raise ValueError('Source or Destination IP not found in device mapping.')

        source = device_mapping[source_ip]
        destination = device_mapping[destination_ip]

        edge_index = torch.tensor([[source], [destination]], dtype=torch.long).to(device)
        edge_attr = torch.tensor([new_flow['features']], dtype=torch.float).to(device)  # Ensure features are in correct order and size

        # Prepare node features
        node_features = torch.zeros((num_nodes, FEATURE_SIZE)).to(device)  # Replace with actual features if available

        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(node_features, edge_index, edge_attr, None, memory)
            anomaly_score = out.item()
            is_anomaly = anomaly_score > 0.5
        return is_anomaly, anomaly_score

    # Example usage of predict_anomaly
    new_flow_example = {
        'source_ip': '198.18.134.99',
        'destination_ip': '198.18.134.2',
        'features': [0.5, 1.2] + [0.0] * (FEATURE_SIZE - 2)  # Replace with actual feature values
    }

    is_anomaly, score = predict_anomaly(new_flow_example, model, memory, device)
    print(f'Anomaly: {is_anomaly}, Score: {score}')

if __name__ == '__main__':
    main()
