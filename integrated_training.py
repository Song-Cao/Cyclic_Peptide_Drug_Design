import torch
from mol_peptide_data_querying import main as preprocess_data
from contrastive_model import train_contrastive_model, MolecularEncoder, ProteinEncoder
from diffusion_model import train_diffusion_model, DiffusionModel
from RL_framework import train_reinforcement_model, PolicyNetwork, RewardPredictor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def load_processed_data(molecular_file, protein_file):
    """
    Load processed molecular and protein data.
    Args:
        molecular_file (str): Path to the molecular data CSV file.
        protein_file (str): Path to the protein data CSV file.
    Returns:
        molecular_features, protein_features (numpy arrays)
    """
    molecular_features = pd.read_csv(molecular_file).iloc[:, 1:].values  # Exclude identifier column
    protein_features = pd.read_csv(protein_file).iloc[:, 1:].values
    return molecular_features, protein_features


def generate_embeddings(encoder, features, batch_size=64):
    """
    Generate embeddings using the provided encoder.
    Args:
        encoder (torch.nn.Module): Trained encoder model.
        features (numpy array): Input features to encode.
        batch_size (int): Batch size for processing.
    Returns:
        embeddings (numpy array): Encoded embeddings.
    """
    dataloader = DataLoader(features, batch_size=batch_size, shuffle=False)
    encoder.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = torch.tensor(batch, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_embeddings = encoder(batch).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def main():
    # Step 1: Data Preprocessing
    print("Step 1: Preprocessing data...")
    preprocess_data()

    # Load processed data
    molecular_file = "cleaned_molecular_features.csv"
    protein_file = "cleaned_protein_features.csv"
    molecular_features, protein_features = load_processed_data(molecular_file, protein_file)

    # Step 2: Contrastive Learning
    print("Step 2: Training contrastive model...")
    input_dim = molecular_features.shape[1]
    latent_dim = 64
    batch_size = 64

    # Train the contrastive model
    dataloader = DataLoader(
        list(zip(protein_features, molecular_features)), batch_size=batch_size, shuffle=True
    )
    protein_encoder = ProteinEncoder(input_dim, latent_dim)
    molecular_encoder = MolecularEncoder(input_dim, latent_dim)

    protein_encoder, molecular_encoder = train_contrastive_model(
        dataloader, input_dim, latent_dim, epochs=20
    )

    # Generate embeddings
    print("Generating molecular and protein embeddings...")
    molecular_embeddings = generate_embeddings(molecular_encoder, molecular_features)
    protein_embeddings = generate_embeddings(protein_encoder, protein_features)

    # Step 3: Diffusion Model Training
    print("Step 3: Training diffusion model...")
    combined_embeddings = np.hstack((molecular_embeddings, protein_embeddings))
    hidden_dim = 128
    timesteps = 100
    diffusion_model = train_diffusion_model(combined_embeddings, latent_dim * 2, hidden_dim, timesteps)

    # Step 4: Reinforcement Learning
    print("Step 4: Training reinforcement model...")
    policy_net = PolicyNetwork(latent_dim * 2, hidden_dim, latent_dim * 2)
    reward_predictor = RewardPredictor(latent_dim * 2, hidden_dim)
    optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(reward_predictor.parameters()), lr=0.001)

    train_reinforcement_model(policy_net, reward_predictor, combined_embeddings, optimizer, gamma=0.99, epochs=50)
    print("Integrated model training complete.")


if __name__ == "__main__":
    main()