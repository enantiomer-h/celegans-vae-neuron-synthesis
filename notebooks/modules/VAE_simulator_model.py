#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NeuronDataset(Dataset):

    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()


        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)


        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, input_dim)

    def encode(self, x):

        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):

        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return self.fc_out(h)

    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var


class VAE_Flexible(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim1=256, hidden_dim2=128):
        super(VAE_Flexible, self).__init__()


        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2


        if hidden_dim2 >= hidden_dim1:
            raise ValueError(
                f"hidden_dim2 ({hidden_dim2}) must be less than hidden_dim1 ({hidden_dim1})"
            )


        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mean = nn.Linear(hidden_dim2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim2, latent_dim)


        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc_out = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):

        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):

        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return self.fc_out(h)

    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def get_compression_ratio(self):

        return self.input_dim / self.latent_dim

    def print_architecture(self):

        print(f"\nVAE_Flexible Architecture:")
        print(f"  Input dimension:     {self.input_dim}")
        print(f"  Hidden layer 1:      {self.hidden_dim1}")
        print(f"  Hidden layer 2:      {self.hidden_dim2}")
        print(f"  Latent dimension:    {self.latent_dim}")
        print(f"  Compression ratio:   {self.get_compression_ratio():.1f}:1")
        print(f"  Total parameters:    {sum(p.numel() for p in self.parameters()):,}")
