#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Optional, Any
import copy

from VAE_simulator_model import VAE, VAE_Flexible, NeuronDataset
from VAE_simulator_model_v2 import VAE_Sigma


def set_random_seed(seed: int = 42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


set_random_seed(42)


def compute_variance_ratio(x_original, x_reconstructed, mask):

    epsilon = 1e-8


    x_masked = x_original * mask
    num_observed = torch.sum(mask, dim=0) + epsilon
    mean_orig = torch.sum(x_masked, dim=0) / num_observed
    var_orig = torch.sum(mask * (x_original - mean_orig) ** 2, dim=0) / num_observed


    x_recon_masked = x_reconstructed * mask
    mean_recon = torch.sum(x_recon_masked, dim=0) / num_observed
    var_recon = torch.sum(mask * (x_reconstructed - mean_recon) ** 2, dim=0) / num_observed


    ratio = var_recon / (var_orig + epsilon)
    return torch.mean(ratio).item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_coefficient: float
) -> Tuple[float, float, float, float]:

    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_variance_ratio = 0.0

    for batch_data, batch_mask in dataloader:
        batch_data = batch_data.to(device)
        batch_mask = batch_mask.to(device)

        optimizer.zero_grad()
        x_hat, mu, log_var = model(batch_data)


        reconstruction_loss = torch.sum(((batch_data - x_hat) * batch_mask) ** 2, dim=1)
        epsilon = 1e-8
        num_observed = torch.sum(batch_mask, dim=1) + epsilon
        reconstruction_loss = reconstruction_loss / num_observed


        kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)


        loss = torch.mean(reconstruction_loss + beta_coefficient * kl_loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += torch.mean(reconstruction_loss).item()
        total_kl_loss += torch.mean(kl_loss).item()


        variance_ratio = compute_variance_ratio(batch_data, x_hat, batch_mask)
        total_variance_ratio += variance_ratio

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_variance_ratio = total_variance_ratio / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss, avg_variance_ratio


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta_coefficient: float
) -> Tuple[float, float, float, float]:

    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_variance_ratio = 0.0

    with torch.no_grad():
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)

            x_hat, mu, log_var = model(batch_data)


            reconstruction_loss = torch.sum(((batch_data - x_hat) * batch_mask) ** 2, dim=1)
            epsilon = 1e-8
            num_observed = torch.sum(batch_mask, dim=1) + epsilon
            reconstruction_loss = reconstruction_loss / num_observed


            kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)


            loss = torch.mean(reconstruction_loss + beta_coefficient * kl_loss)

            total_loss += loss.item()
            total_recon_loss += torch.mean(reconstruction_loss).item()
            total_kl_loss += torch.mean(kl_loss).item()


            variance_ratio = compute_variance_ratio(batch_data, x_hat, batch_mask)
            total_variance_ratio += variance_ratio

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_variance_ratio = total_variance_ratio / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss, avg_variance_ratio


class EarlyStopping:


    def __init__(self, patience: int = 10, min_delta: float = 0.0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model: nn.Module):

        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_vae_with_validation(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    beta_coefficient: float,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Tuple[Dict[str, list], nn.Module]:

    logger = logging.getLogger(__name__)
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_variance_ratio': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_variance_ratio': []
    }

    pbar = tqdm(range(epochs), desc="Training Progress", disable=not verbose)

    for epoch in pbar:

        train_loss, train_recon, train_kl, train_var_ratio = train_epoch(
            model, train_dataloader, optimizer, device, beta_coefficient
        )


        val_loss, val_recon, val_kl, val_var_ratio = validate_epoch(
            model, val_dataloader, device, beta_coefficient
        )


        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon)
        history['train_kl_loss'].append(train_kl)
        history['train_variance_ratio'].append(train_var_ratio)
        history['val_loss'].append(val_loss)
        history['val_recon_loss'].append(val_recon)
        history['val_kl_loss'].append(val_kl)
        history['val_variance_ratio'].append(val_var_ratio)


        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_var_ratio': f'{val_var_ratio:.4f}'
        })


        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break


    early_stopping.load_best_model(model)
    logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")

    return history, model


def objective_v2(
    trial: optuna.Trial,
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    X_val: torch.Tensor,
    mask_val: torch.Tensor,
    num_features: int,
    device: torch.device,
    use_flexible: bool = True
) -> float:


    latent_dim = trial.suggest_int('latent_dim', 36, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    beta_coefficient = trial.suggest_float('beta_coefficient', 0.4, 2.0, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 100)

    if use_flexible:

        hidden_dim1 = trial.suggest_categorical('hidden_dim1', [128, 256, 512, 1024])
        hidden_dim2 = trial.suggest_categorical('hidden_dim2', [64, 128, 256, 512])


        if hidden_dim2 >= hidden_dim1:
            return float('inf')


    train_dataset = NeuronDataset(X_train, mask_train)
    val_dataset = NeuronDataset(X_val, mask_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    if use_flexible:
        model = VAE_Flexible(
            input_dim=num_features,
            latent_dim=latent_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2
        ).to(device)
    else:
        model = VAE(input_dim=num_features, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    model.train()
    best_val_loss = float('inf')

    for epoch in range(epochs):

        train_epoch(model, train_dataloader, optimizer, device, beta_coefficient)


        val_loss, _, _, _ = validate_epoch(model, val_dataloader, device, beta_coefficient)


        if val_loss < best_val_loss:
            best_val_loss = val_loss


        trial.report(val_loss, epoch)


        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def main(
    num_features: int,
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    X_val: torch.Tensor,
    mask_val: torch.Tensor,
    optuna_seed: int = 42,
    optuna_process: bool = False,
    optuna_trials: int = 100,
    use_flexible: bool = True,
    hidden_dim1: Optional[int] = None,
    hidden_dim2: Optional[int] = None,
    latent_dim: Optional[int] = None,
    beta_coefficient: Optional[float] = None,
    early_stopping_patience: int = 10,
    device: str = 'cuda'
) -> Tuple[Dict[str, Any], nn.Module, Dict[str, list]]:

    logger = logging.getLogger(__name__)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if optuna_process:
        logger.info("Running Optuna hyperparameter optimization...")


        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=optuna_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )


        study.optimize(
            lambda trial: objective_v2(
                trial, X_train, mask_train, X_val, mask_val,
                num_features, device, use_flexible
            ),
            n_trials=optuna_trials,
            show_progress_bar=True
        )

        logger.info("Optuna optimization complete")
        logger.info(f"Best hyperparameters: {study.best_params}")


        try:
            import optuna.visualization as vis
            vis.plot_optimization_history(study).show()
            vis.plot_param_importances(study).show()
            vis.plot_parallel_coordinate(study).show()
        except Exception as e:
            (f'{e}',)

        best_params = study.best_params

    else:
        logger.info("Using pre-specified hyperparameters...")


        if use_flexible:
            best_params = {
                'latent_dim': latent_dim if latent_dim is not None else 100,
                'learning_rate': 0.001,
                'beta_coefficient': beta_coefficient if beta_coefficient is not None else 0.5,
                'batch_size': 32,
                'epochs': 100,
                'hidden_dim1': hidden_dim1 or 256,
                'hidden_dim2': hidden_dim2 or 128
            }
        else:
            best_params = {
                'latent_dim': latent_dim if latent_dim is not None else 100,
                'learning_rate': 0.001,
                'beta_coefficient': beta_coefficient if beta_coefficient is not None else 0.5,
                'batch_size': 32,
                'epochs': 100
            }

        logger.info(f"Parameters: {best_params}")


    train_dataset = NeuronDataset(X_train, mask_train)
    val_dataset = NeuronDataset(X_val, mask_val)
    train_dataloader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)


    if use_flexible:
        best_model = VAE_Flexible(
            input_dim=num_features,
            latent_dim=best_params['latent_dim'],
            hidden_dim1=best_params['hidden_dim1'],
            hidden_dim2=best_params['hidden_dim2']
        ).to(device)
        best_model.print_architecture()
    else:
        best_model = VAE(
            input_dim=num_features,
            latent_dim=best_params['latent_dim']
        ).to(device)


    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])


    logger.info("Training final model with validation...")
    history, best_model = train_vae_with_validation(
        best_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        best_params['epochs'],
        device,
        best_params['beta_coefficient'],
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )

    logger.info("Training complete")

    return best_params, best_model, history


def sigma_vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    log_sigma: torch.Tensor,
    mu_z: torch.Tensor,
    log_var_z: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    lambda_var: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    epsilon = 1e-8


    if log_sigma.dim() == 1:
        log_sigma = log_sigma.unsqueeze(0).expand_as(x)


    sigma = torch.exp(log_sigma) + epsilon
    sigma_sq = sigma ** 2


    squared_error = ((x - x_hat) * mask) ** 2
    recon_per_feature = 0.5 * squared_error / sigma_sq + log_sigma * mask


    num_observed = torch.sum(mask, dim=1) + epsilon
    recon_loss = torch.sum(recon_per_feature, dim=1) / num_observed


    kl_loss = -0.5 * torch.sum(1 + log_var_z - mu_z ** 2 - log_var_z.exp(), dim=1)


    var_loss = torch.tensor(0.0, device=x.device)
    if lambda_var > 0:

        x_masked = x * mask
        x_hat_masked = x_hat * mask


        n_obs_per_feature = torch.sum(mask, dim=0) + epsilon
        x_mean = torch.sum(x_masked, dim=0) / n_obs_per_feature
        x_hat_mean = torch.sum(x_hat_masked, dim=0) / n_obs_per_feature

        x_var = torch.sum(mask * (x - x_mean.unsqueeze(0)) ** 2, dim=0) / n_obs_per_feature
        x_hat_var = torch.sum(mask * (x_hat - x_hat_mean.unsqueeze(0)) ** 2, dim=0) / n_obs_per_feature


        var_loss = torch.mean((torch.log(x_hat_var + epsilon) - torch.log(x_var + epsilon)) ** 2)


    total_loss = torch.mean(recon_loss + beta * kl_loss) + lambda_var * var_loss

    return total_loss, torch.mean(recon_loss), torch.mean(kl_loss), var_loss


def get_cyclical_beta(epoch: int, cycle_length: int = 20, n_cycles: int = 4, beta_max: float = 1.0) -> float:

    if epoch >= n_cycles * cycle_length:
        return beta_max

    position_in_cycle = epoch % cycle_length
    ramp_length = cycle_length // 2

    if position_in_cycle < ramp_length:
        return (position_in_cycle / ramp_length) * beta_max
    return beta_max


def get_monotonic_beta(epoch: int, warmup_epochs: int = 20, beta_max: float = 1.0) -> float:

    if epoch >= warmup_epochs:
        return beta_max
    return (epoch / warmup_epochs) * beta_max


def train_epoch_sigma_vae(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    lambda_var: float = 0.0
) -> Tuple[float, float, float, float, float]:

    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_var_loss = 0.0
    total_variance_ratio = 0.0
    n_batches = 0

    for batch_data, batch_mask in dataloader:
        batch_data = batch_data.to(device)
        batch_mask = batch_mask.to(device)

        optimizer.zero_grad()


        x_hat, mu_z, log_var_z = model(batch_data)


        loss, recon_loss, kl_loss, var_loss = sigma_vae_loss(
            batch_data, x_hat, model.log_sigma_decoder,
            mu_z, log_var_z, batch_mask, beta, lambda_var
        )


        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_var_loss += var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss
        total_variance_ratio += compute_variance_ratio(batch_data, x_hat, batch_mask)
        n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kl / n_batches,
        total_var_loss / n_batches,
        total_variance_ratio / n_batches
    )


def validate_epoch_sigma_vae(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    lambda_var: float = 0.0
) -> Tuple[float, float, float, float, float]:

    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_var_loss = 0.0
    total_variance_ratio = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)

            x_hat, mu_z, log_var_z = model(batch_data)

            loss, recon_loss, kl_loss, var_loss = sigma_vae_loss(
                batch_data, x_hat, model.log_sigma_decoder,
                mu_z, log_var_z, batch_mask, beta, lambda_var
            )

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_var_loss += var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss
            total_variance_ratio += compute_variance_ratio(batch_data, x_hat, batch_mask)
            n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kl / n_batches,
        total_var_loss / n_batches,
        total_variance_ratio / n_batches
    )


def train_sigma_vae_with_annealing(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    beta_max: float = 1.0,
    annealing_type: str = 'cyclical',
    cycle_length: int = 20,
    n_cycles: int = 4,
    early_stopping_patience: int = 10,
    lambda_var: float = 0.0,
    verbose: bool = True
) -> Tuple[Dict[str, list], nn.Module]:

    logger = logging.getLogger(__name__)

    if lambda_var > 0:
        logger.info(f"Variance preservation loss enabled with lambda_var={lambda_var}")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_kl_loss': [], 'val_kl_loss': [],
        'train_var_loss': [], 'val_var_loss': [],
        'train_variance_ratio': [], 'val_variance_ratio': [],
        'beta': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc="Training Sigma-VAE") if verbose else range(epochs)

    for epoch in pbar:

        if annealing_type == 'cyclical':
            beta = get_cyclical_beta(epoch, cycle_length, n_cycles, beta_max)
        elif annealing_type == 'monotonic':
            beta = get_monotonic_beta(epoch, cycle_length, beta_max)
        else:
            beta = beta_max


        train_loss, train_recon, train_kl, train_var_loss, train_var_ratio = train_epoch_sigma_vae(
            model, train_dataloader, optimizer, device, beta, lambda_var
        )


        val_loss, val_recon, val_kl, val_var_loss, val_var_ratio = validate_epoch_sigma_vae(
            model, val_dataloader, device, beta, lambda_var
        )


        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon_loss'].append(train_recon)
        history['val_recon_loss'].append(val_recon)
        history['train_kl_loss'].append(train_kl)
        history['val_kl_loss'].append(val_kl)
        history['train_var_loss'].append(train_var_loss)
        history['val_var_loss'].append(val_var_loss)
        history['train_variance_ratio'].append(train_var_ratio)
        history['val_variance_ratio'].append(val_var_ratio)
        history['beta'].append(beta)


        if verbose:
            postfix = {
                'β': f'{beta:.3f}',
                'train': f'{train_loss:.4f}',
                'val': f'{val_loss:.4f}',
                'var_r': f'{val_var_ratio:.3f}'
            }
            if lambda_var > 0:
                postfix['var_L'] = f'{val_var_loss:.4f}'
            pbar.set_postfix(postfix)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, model


def main_sigma_vae(
    num_features: int,
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    X_val: torch.Tensor,
    mask_val: torch.Tensor,
    optuna_seed: int = 42,
    hidden_dim1: int = 256,
    hidden_dim2: int = 128,
    latent_dim: int = 64,
    beta_max: float = 1.0,
    annealing_type: str = 'cyclical',
    cycle_length: int = 20,
    n_cycles: int = 4,
    sigma_init: float = -2.0,
    early_stopping_patience: int = 10,
    lambda_var: float = 0.0,
    device: str = 'cuda'
) -> Tuple[Dict[str, Any], nn.Module, Dict[str, list]]:

    logger = logging.getLogger(__name__)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')


    set_random_seed(optuna_seed)

    logger.info("Initializing Sigma-VAE model...")


    if annealing_type == 'cyclical':
        epochs = n_cycles * cycle_length + 20
    else:
        epochs = 100

    best_params = {
        'latent_dim': latent_dim,
        'learning_rate': 0.001,
        'beta_max': beta_max,
        'batch_size': 32,
        'epochs': epochs,
        'hidden_dim1': hidden_dim1,
        'hidden_dim2': hidden_dim2,
        'annealing_type': annealing_type,
        'cycle_length': cycle_length,
        'n_cycles': n_cycles,
        'sigma_init': sigma_init,
        'lambda_var': lambda_var
    }

    logger.info(f"Parameters: {best_params}")


    train_dataset = NeuronDataset(X_train, mask_train)
    val_dataset = NeuronDataset(X_val, mask_val)
    train_dataloader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)


    best_model = VAE_Sigma(
        input_dim=num_features,
        latent_dim=latent_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        sigma_init=sigma_init,
        sigma_learnable=True
    ).to(device)

    best_model.print_architecture()


    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])


    logger.info(f"Training Sigma-VAE with {annealing_type} KL annealing...")
    if lambda_var > 0:
        logger.info(f"Variance preservation loss enabled (lambda_var={lambda_var})")
    history, best_model = train_sigma_vae_with_annealing(
        model=best_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        beta_max=beta_max,
        annealing_type=annealing_type,
        cycle_length=cycle_length,
        n_cycles=n_cycles,
        early_stopping_patience=early_stopping_patience,
        lambda_var=lambda_var,
        verbose=True
    )

    logger.info("Sigma-VAE training complete")


    sigma = best_model.get_learned_sigma()
    logger.info(f"Learned decoder σ: mean={sigma.mean():.4f}, "
               f"min={sigma.min():.4f}, max={sigma.max():.4f}")

    return best_params, best_model, history


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)


    n_train = 80
    n_val = 20
    n_features = 2040

    X_train = torch.randn(n_train, n_features)
    mask_train = torch.rand(n_train, n_features) > 0.3
    mask_train = mask_train.float()

    X_val = torch.randn(n_val, n_features)
    mask_val = torch.rand(n_val, n_features) > 0.3
    mask_val = mask_val.float()


    best_params, best_model, history = main(
        num_features=n_features,
        X_train=X_train,
        mask_train=mask_train,
        X_val=X_val,
        mask_val=mask_val,
        optuna_process=False,
        use_flexible=True,
        hidden_dim1=256,
        hidden_dim2=128,
        device='cpu'
    )

    print(f"\nBest parameters: {best_params}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
