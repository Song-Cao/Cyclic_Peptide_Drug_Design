import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class PeptideDataset(Dataset):
    """
    Custom dataset for loading peptide features.
    """
    def __init__(self, file_path):
        """
        Initialize dataset by loading data from a CSV file.
        Args:
            file_path (str): Path to the CSV file containing peptide features.
        """
        self.data = pd.read_csv(file_path)
        self.features = self.data.iloc[:, 1:].values  # Exclude identifier column

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return torch.softmax(self.fc3(x), dim=-1)


class RewardPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


def calculate_reward(peptides, reward_predictor):
    """
    Compute rewards for generated peptides.
    Rewards are shaped by structural and functional criteria.
    """
    rewards = []
    for peptide in peptides:
        peptide_tensor = peptide.unsqueeze(0)
        predicted_reward = reward_predictor(peptide_tensor).item()
        # Example: penalize invalid values (can be customized for real use cases)
        if torch.isnan(predicted_reward) or predicted_reward < 0:
            predicted_reward = -10
        rewards.append(predicted_reward)
    return torch.tensor(rewards, dtype=torch.float32)


def train_reinforcement_model(policy_net, reward_predictor, dataloader, optimizer, gamma=0.99, epochs=50, baseline_weight=0.9):
    """
    Train the policy network using policy gradients with a reward baseline.
    """
    baseline = 0  # Initialize reward baseline

    for epoch in range(epochs):
        log_probs = []
        rewards = []
        optimizer.zero_grad()

        # Simulate generation and reward collection
        for peptides in dataloader:
            for peptide in peptides:
                action_probs = policy_net(peptide)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                reward = reward_predictor(peptide.unsqueeze(0)).item()

                log_probs.append(log_prob)
                rewards.append(reward)

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Update baseline for variance reduction
        baseline = baseline_weight * baseline + (1 - baseline_weight) * discounted_rewards.mean()

        # Policy gradient loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            adjusted_reward = reward - baseline  # Subtract baseline
            policy_loss.append(-log_prob * adjusted_reward)
        policy_loss = torch.stack(policy_loss).sum()

        # Backpropagation
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {policy_loss.item()}, Baseline: {baseline.item()}")


def main():
    # Load data
    file_path = "cleaned_cyclic_peptide_data.csv"
    dataset = PeptideDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize models and optimizer
    input_dim = dataset[0].shape[0]
    hidden_dim = 256
    output_dim = input_dim  # Assume output dimensions match input dimensions

    print("Initializing policy network and reward predictor...")
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    reward_predictor = RewardPredictor(input_dim, hidden_dim)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(reward_predictor.parameters()), lr=0.001)

    # Train reinforcement model
    print("Training reinforcement model...")
    train_reinforcement_model(policy_net, reward_predictor, dataloader, optimizer)
    print("Reinforcement learning complete.")


if __name__ == "__main__":
    main()