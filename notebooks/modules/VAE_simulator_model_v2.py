#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.utils.data import Dataset


from VAE_simulator_model import VAE, VAE_Flexible, NeuronDataset


class VAE_Sigma(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim1=256, hidden_dim2=128,
                 sigma_init=-2.0, sigma_learnable=True):
        super(VAE_Sigma, self).__init__()


        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.sigma_learnable = sigma_learnable


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
        self.fc_out_mean = nn.Linear(hidden_dim1, input_dim)


        if sigma_learnable:
            self.log_sigma_decoder = nn.Parameter(
                torch.full((input_dim,), sigma_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_sigma_decoder',
                torch.full((input_dim,), sigma_init, dtype=torch.float32)
            )

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
        return self.fc_out_mean(h)

    def decode_with_sigma(self, z):

        mu_x = self.decode(z)
        return mu_x, self.log_sigma_decoder

    def forward(self, x):

        mu_z, log_var_z = self.encode(x)
        z = self.reparameterize(mu_z, log_var_z)
        x_hat = self.decode(z)
        return x_hat, mu_z, log_var_z

    def sample(self, num_samples, device='cpu'):

        self.eval()
        with torch.no_grad():

            z = torch.randn(num_samples, self.latent_dim, device=device)


            mu_x = self.decode(z)


            sigma_x = torch.exp(self.log_sigma_decoder).to(device)


            epsilon = torch.randn_like(mu_x)
            samples = mu_x + sigma_x.unsqueeze(0) * epsilon

        return samples

    def sample_with_variance_matching(
        self,
        num_samples: int,
        target_variance: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:

        self.eval()
        with torch.no_grad():

            z = torch.randn(num_samples, self.latent_dim, device=device)


            mu_x = self.decode(z)


            current_var = torch.var(mu_x, dim=0) + 1e-8


            if not isinstance(target_variance, torch.Tensor):
                target_variance = torch.tensor(target_variance, device=device)
            else:
                target_variance = target_variance.to(device)


            scale = torch.sqrt(target_variance / current_var)


            mean_x = mu_x.mean(dim=0)
            centered_mu = mu_x - mean_x
            scaled_mu = centered_mu * scale.unsqueeze(0) + mean_x


            sigma_x = torch.exp(self.log_sigma_decoder).to(device)
            epsilon = torch.randn_like(scaled_mu)
            samples = scaled_mu + sigma_x.unsqueeze(0) * epsilon

        return samples

    def get_learned_sigma(self):

        return torch.exp(self.log_sigma_decoder).detach().cpu().numpy()

    def get_compression_ratio(self):

        return self.input_dim / self.latent_dim

    def print_architecture(self):

        sigma_mean = torch.exp(self.log_sigma_decoder).mean().item()
        print(f"\nVAE_Sigma Architecture:")
        print(f"  Input dimension:     {self.input_dim}")
        print(f"  Hidden layer 1:      {self.hidden_dim1}")
        print(f"  Hidden layer 2:      {self.hidden_dim2}")
        print(f"  Latent dimension:    {self.latent_dim}")
        print(f"  Compression ratio:   {self.get_compression_ratio():.1f}:1")
        print(f"  Sigma learnable:     {self.sigma_learnable}")
        print(f"  Mean decoder σ:      {sigma_mean:.4f}")
        print(f"  Total parameters:    {sum(p.numel() for p in self.parameters()):,}")
