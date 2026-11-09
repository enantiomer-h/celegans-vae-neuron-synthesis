
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import optuna
from tqdm import tqdm


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def objective(trial):

    latent_dim = trial.suggest_int('latent_dim', 5, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    beta_coefficient = trial.suggest_uniform('beta_coefficient', 0.0001, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 100)


    dataset = NeuronDataset(X_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_dim=num_features, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch_data)

            reconstruction_loss = torch.sum(((batch_data - x_hat) * batch_mask) ** 2, dim=1)

            epsilon = 1e-8
            reconstruction_loss = torch.sum(((batch_data - x_hat) * batch_mask) ** 2, dim=1)
            num_observed = torch.sum(batch_mask, dim=1) + epsilon

            kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

            loss = torch.mean(reconstruction_loss / num_observed + beta_coefficient * kl_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss

def main(num_features, optuna_seed, optuna_process, X_tensor, mask_tensor):
    if optuna_process == True:

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=optuna_seed))
        study.optimize(objective, n_trials=100)


        print("Best hyperparameters: ", study.best_params)

    if optuna_process == True:

        optuna.visualization.plot_optimization_history(study).show()


        optuna.visualization.plot_param_importances(study).show()


        optuna.visualization.plot_parallel_coordinate(study).show()


        optuna.visualization.plot_slice(study).show()


        optuna.visualization.plot_contour(study).show()


        optuna.visualization.plot_edf(study).show()

    if 'study' in globals():
        print("After running the optimization:")
        best_params = study.best_params
    else:
        print("Without running the optimization:")
        best_params = {'latent_dim': 50, 'learning_rate': 0.001, 'beta_coefficient': 0.01, 'batch_size': 32, 'epochs': 100}


    dataset = NeuronDataset(X_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = VAE(input_dim=num_features, latent_dim=best_params['latent_dim']).to(device)


    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])


    epochs = best_params['epochs']
    beta_coefficient = best_params['beta_coefficient']

    best_model.train()

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        total_loss = 0
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = best_model(batch_data)
            reconstruction_loss = torch.sum(((batch_data - x_hat) * batch_mask) ** 2, dim=1)
            epsilon = 1e-8
            num_observed = torch.sum(batch_mask, dim=1) + epsilon
            kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            loss = torch.mean(reconstruction_loss / num_observed + beta_coefficient * kl_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)


    return best_params, best_model
