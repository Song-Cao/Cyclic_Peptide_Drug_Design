import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class PeptideDataset(Dataset):
    """
    Custom dataset for loading peptide features.
    """
    def __init__(self, molecular_file, protein_file):
        """
        Initialize dataset by loading molecular and protein data from CSV files.
        Args:
            molecular_file (str): Path to the molecular features CSV file.
            protein_file (str): Path to the protein features CSV file.
        """
        self.molecular_data = pd.read_csv(molecular_file).iloc[:, 1:].values  # Exclude identifier column
        self.protein_data = pd.read_csv(protein_file).iloc[:, 1:].values

        if len(self.molecular_data) != len(self.protein_data):
            raise ValueError("Molecular and protein data must have the same number of samples.")

    def __len__(self):
        return len(self.molecular_data)

    def __getitem__(self, idx):
        molecular_features = torch.tensor(self.molecular_data[idx], dtype=torch.float32)
        protein_features = torch.tensor(self.protein_data[idx], dtype=torch.float32)
        combined_features = torch.cat((molecular_features, protein_features), dim=0)
        return combined_features


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        # Additional layers for hierarchical processing
        self.residual_fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        # Add time embedding
        t_embed = self.time_embedding(t, x.size(-1))
        x = x + t_embed

        # Forward pass with residual connection
        x = F.relu(self.bn1(self.fc1(x)))
        res = self.residual_fc(x)
        x = F.relu(self.bn2(self.fc2(x + res)))
        x = self.dropout(x)

        return self.fc3(x)

    def time_embedding(self, t, dim):
        """Enhanced sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def forward_diffusion(x_0, alpha_t, timesteps):
    """Simulate forward diffusion process."""
    batch_size = x_0.size(0)
    t = torch.randint(0, timesteps, (batch_size,), device=x_0.device).long()
    alpha_bar = torch.cumprod(alpha_t[t], dim=0)
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar)[:, None] * x_0 + torch.sqrt(1 - alpha_bar)[:, None] * noise
    return x_t, noise, t


def diffusion_loss(model, x_0, alpha_t, timesteps):
    x_t, noise, t = forward_diffusion(x_0, alpha_t, timesteps)
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)


def train_diffusion_model(dataloader, input_dim, hidden_dim, timesteps, epochs=20, learning_rate=1e-3):
    model = DiffusionModel(input_dim, hidden_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    alpha_t = torch.linspace(0.99, 0.001, timesteps).to('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            loss = diffusion_loss(model, batch, alpha_t, timesteps)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model


def main():
    # File paths for molecular and protein data
    molecular_file = "molecular_features.csv"
    protein_file = "protein_features.csv"

    # Load dataset
    dataset = PeptideDataset(molecular_file, protein_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Determine dimensions
    input_dim = dataset[0].shape[0]
    hidden_dim = 256
    timesteps = 200

    # Train the diffusion model
    print("Training diffusion model...")
    model = train_diffusion_model(dataloader, input_dim, hidden_dim, timesteps)
    print("Diffusion model training complete.")


if __name__ == "__main__":
    main()