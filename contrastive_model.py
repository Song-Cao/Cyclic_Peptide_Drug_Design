import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define dataset class
class PeptideDataset(Dataset):
    def __init__(self, protein_file, molecular_file):
        """
        Initialize dataset by loading data from protein and molecular feature files.
        Args:
            protein_file (str): Path to the protein features CSV file.
            molecular_file (str): Path to the molecular features CSV file.
        """
        self.protein_features = pd.read_csv(protein_file).iloc[:, 1:].values  # Exclude identifier column
        self.molecular_features = pd.read_csv(molecular_file).iloc[:, 1:].values  # Exclude identifier column

        if len(self.protein_features) != len(self.molecular_features):
            raise ValueError("Protein and molecular features must have the same number of samples.")

    def __len__(self):
        return len(self.protein_features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.protein_features[idx], dtype=torch.float32),
            torch.tensor(self.molecular_features[idx], dtype=torch.float32),
        )

# Hierarchical molecular encoder
class MolecularEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MolecularEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.normalize(self.fc3(x), p=2, dim=1)
        return x

# Hierarchical protein encoder
class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ProteinEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.normalize(self.fc3(x), p=2, dim=1)
        return x

# Expanded contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_protein, z_molecular):
        batch_size = z_protein.size(0)
        labels = torch.arange(batch_size).to(z_protein.device)

        # Inter-modal similarity
        logits = torch.mm(z_protein, z_molecular.T) / self.temperature
        loss_inter = F.cross_entropy(logits, labels)

        # Intra-modal similarity
        protein_self_logits = torch.mm(z_protein, z_protein.T) / self.temperature
        molecular_self_logits = torch.mm(z_molecular, z_molecular.T) / self.temperature
        loss_intra_protein = F.cross_entropy(protein_self_logits, labels)
        loss_intra_molecular = F.cross_entropy(molecular_self_logits, labels)

        return loss_inter + 0.5 * (loss_intra_protein + loss_intra_molecular)

# Training function
def train_contrastive_model(dataloader, input_dim, latent_dim, epochs=20, learning_rate=0.001):
    protein_encoder = ProteinEncoder(input_dim, latent_dim)
    molecular_encoder = MolecularEncoder(input_dim, latent_dim)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(list(protein_encoder.parameters()) + list(molecular_encoder.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    protein_encoder.train()
    molecular_encoder.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for protein_batch, molecular_batch in dataloader:
            optimizer.zero_grad()
            z_protein = protein_encoder(protein_batch)
            z_molecular = molecular_encoder(molecular_batch)
            loss = criterion(z_protein, z_molecular)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return protein_encoder, molecular_encoder

def main():
    # File paths for protein and molecular features
    protein_file = "protein_features.csv"
    molecular_file = "molecular_features.csv"

    # Load dataset and dataloader
    dataset = PeptideDataset(protein_file, molecular_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Determine input dimensions
    input_dim = dataset[0][0].shape[0]  # Protein feature dimension
    latent_dim = 64

    print("Training contrastive model...")
    protein_encoder, molecular_encoder = train_contrastive_model(
        dataloader, input_dim, latent_dim
    )
    print("Training complete.")

if __name__ == "__main__":
    main()