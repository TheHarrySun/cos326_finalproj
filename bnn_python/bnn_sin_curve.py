#!/usr/bin/env python3
"""
Bayesian Neural Network for Sin Curve Regression
Implements a BNN using variational inference with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty
    Uses mean-field variational inference (Gaussian posterior)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters (mean and log variance)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Prior parameters (standard normal prior)
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -3.0)  # Small initial variance
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -3.0)
    
    def forward(self, x, sample=True):
        """
        Forward pass with optional weight sampling
        
        Args:
            x: Input tensor
            sample: If True, sample weights from posterior; if False, use mean
        """
        if sample:
            # Sample weights using reparameterization trick
            weight_sigma = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_eps
            
            # Sample biases
            bias_sigma = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            # Use mean weights (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between posterior q(w) and prior p(w)
        KL(q(w) || p(w)) for Gaussian distributions
        """
        # Weight KL
        weight_sigma = torch.exp(0.5 * self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_sigma**2 +
            weight_sigma**2 / self.prior_sigma**2 -
            1.0 - self.weight_logvar +
            2 * math.log(self.prior_sigma)
        )
        
        # Bias KL
        bias_sigma = torch.exp(0.5 * self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_sigma**2 +
            bias_sigma**2 / self.prior_sigma**2 -
            1.0 - self.bias_logvar +
            2 * math.log(self.prior_sigma)
        )
        
        return weight_kl + bias_kl


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network with 1 hidden layer
    Architecture: input -> 20 -> output
    """
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        super().__init__()
        
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, output_dim)
        
        # Noise precision parameter (learnable observation noise)
        self.log_noise_precision = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, sample=True):
        """Forward pass through network"""
        x = self.layer1(x, sample=sample)
        x = torch.relu(x)
        x = self.layer2(x, sample=sample)
        return x
    
    def kl_divergence(self):
        """Total KL divergence for all layers"""
        return (self.layer1.kl_divergence() + 
                self.layer2.kl_divergence())
    
    def elbo_loss(self, x, y, n_samples=1):
        """
        Compute ELBO (Evidence Lower Bound) loss
        ELBO = E[log p(y|x,w)] - KL[q(w)||p(w)]
        We want to maximize ELBO, so we minimize negative ELBO
        """
        # Negative log likelihood (reconstruction loss)
        outputs = torch.stack([self.forward(x, sample=True) for _ in range(n_samples)])
        output_mean = outputs.mean(dim=0)
        
        noise_precision = torch.exp(self.log_noise_precision)
        nll = 0.5 * noise_precision * torch.sum((output_mean - y)**2)
        nll -= 0.5 * y.numel() * self.log_noise_precision
        nll += 0.5 * y.numel() * math.log(2 * math.pi)
        
        # KL divergence
        kl = self.kl_divergence()
        
        # ELBO loss (negative ELBO)
        loss = nll + kl
        
        return loss, nll, kl


def generate_sin_data(n_train=50, n_test=200, noise_std=0.1):
    """
    Generate synthetic sin curve data with noise
    
    Args:
        n_train: Number of training points
        n_test: Number of test points
        noise_std: Standard deviation of observation noise
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Training data: sparse samples from sin curve
    X_train = np.random.uniform(-5, 5, n_train)
    y_train = np.sin(X_train) + np.random.normal(0, noise_std, n_train)
    
    # Test data: dense grid for visualization
    X_test = np.linspace(-5, 5, n_test)
    y_test = np.sin(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test


def generate_polynomial_data(n_train=50, n_test=200, noise_std=0.25):
    """
    Generate polynomial curve data: y = 2*x^2 + x + 1 + noise
    
    Args:
        n_train: Number of training points
        n_test: Number of test points
        noise_std: Standard deviation of observation noise
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Training data: sparse samples from polynomial curve
    X_train = np.random.uniform(-5, 5, n_train)
    y_train = 2.0 * X_train**2 + X_train + 5.0 * np.random.normal(0, noise_std, n_train)
    
    # Test data: dense grid for visualization
    X_test = np.linspace(-5, 5, n_test)
    y_test = 2.0 * X_test**2 + X_test + 1.0
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test


def generate_linear_data(n_train=50, n_test=200, noise_std=0.2):
    """
    Generate linear regression data: y = 2.5*x - 1.5 + noise
    
    Args:
        n_train: Number of training points
        n_test: Number of test points
        noise_std: Standard deviation of observation noise
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Training data: sparse samples from linear function
    X_train = np.random.uniform(-5, 5, n_train)
    y_train = 2.5 * X_train - 1.5 + np.random.normal(0, noise_std, n_train)
    
    # Test data: dense grid for visualization
    X_test = np.linspace(-6, 6, n_test)
    y_test = 2.5 * X_test - 1.5
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test


def train_bnn(model, X_train, y_train, n_epochs=1000, lr=0.01, n_samples=3):
    """
    Train the Bayesian Neural Network
    
    Args:
        model: BayesianNN model
        X_train: Training inputs
        y_train: Training targets
        n_epochs: Number of training epochs
        lr: Learning rate
        n_samples: Number of samples for Monte Carlo estimation
    
    Returns:
        List of losses per epoch
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    
    print("Training Bayesian Neural Network...")
    print(f"Epochs: {n_epochs}, Learning rate: {lr}, MC samples: {n_samples}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Compute ELBO loss
        loss, nll, kl = model.elbo_loss(X_train, y_train, n_samples=n_samples)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss: {loss.item():8.4f} | "
                  f"NLL: {nll.item():8.4f} | "
                  f"KL: {kl.item():8.4f}")
    
    print("-" * 60)
    print("Training complete!")
    return losses


def predict_with_uncertainty(model, X_test, n_samples=100):
    """
    Make predictions with uncertainty estimates
    
    Args:
        model: Trained BayesianNN
        X_test: Test inputs
        n_samples: Number of forward passes for uncertainty estimation
    
    Returns:
        mean predictions, standard deviations
    """
    model.eval()
    with torch.no_grad():
        predictions = torch.stack([model(X_test, sample=True) for _ in range(n_samples)])
    
    # Compute mean and std across samples
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return mean_pred, std_pred


def plot_results(X_train, y_train, X_test, y_test, mean_pred, std_pred, losses):
    """
    Create visualization plots
    
    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs
        y_test: True test outputs
        mean_pred: Mean predictions
        std_pred: Standard deviation of predictions
        losses: Training losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Predictions with uncertainty
    X_test_np = X_test.numpy().flatten()
    y_test_np = y_test.numpy().flatten()
    mean_np = mean_pred.numpy().flatten()
    std_np = std_pred.numpy().flatten()
    
    # True function
    ax1.plot(X_test_np, y_test_np, 'g-', linewidth=2, label='True Function', zorder=1)
    
    # BNN predictions
    ax1.plot(X_test_np, mean_np, 'b-', linewidth=2, label='BNN Mean', zorder=2)
    
    # Uncertainty bands (±2 std = ~95% confidence)
    ax1.fill_between(X_test_np, 
                     mean_np - 2*std_np, 
                     mean_np + 2*std_np,
                     alpha=0.3, color='blue', label='95% Confidence', zorder=0)
    
    # Training data
    ax1.scatter(X_train.numpy().flatten(), y_train.numpy().flatten(), 
                c='red', s=5, alpha=0.6, label='Training Data', zorder=3)
    
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('y', fontsize=14)
    ax1.set_title('Bayesian Neural Network - Polynomial Curve Prediction', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([X_test_np.min() - 0.5, X_test_np.max() + 0.5])
    ax1.set_ylim([min(y_test_np.min(), mean_np.min()) - 1, max(y_test_np.max(), mean_np.max()) + 1])
    
    # Plot 2: Training loss
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('ELBO Loss', fontsize=14)
    ax2.set_title('Training Loss (Negative ELBO)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(losses)])
    
    plt.tight_layout()
    plt.savefig('bnn_polynomial_curve_results_SGD.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'bnn_sin_curve_results.png'")
    plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("Bayesian Neural Network - Polynomial Curve Regression")
    print("="*60)
    
    # 1. Generate dataset
    print("\n[1] Generating polynomial curve dataset...")
    X_train, y_train, X_test, y_test = generate_polynomial_data(
        n_train=1000, 
        n_test=200, 
        noise_std=0.25
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # 2. Create BNN model
    print("\n[2] Creating Bayesian Neural Network...")
    model = BayesianNN(input_dim=1, hidden_dim=20, output_dim=1)
    print(f"Architecture: 1 -> 20 -> 1")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Train the model
    print("\n[3] Training the model...")
    losses = train_bnn(
        model, 
        X_train, 
        y_train, 
        n_epochs=10000, 
        lr=0.0001,
        n_samples=3
    )
    
    # 4. Make predictions with uncertainty
    print("\n[4] Making predictions with uncertainty estimation...")
    mean_pred, std_pred = predict_with_uncertainty(model, X_test, n_samples=100)
    
    # Compute metrics
    mse = torch.mean((mean_pred - y_test)**2).item()
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Mean uncertainty (std): {std_pred.mean().item():.4f}")
    
    # Print trained model weights
    print("\n" + "="*60)
    print("Trained Model Weights (All Parameters)")
    print("="*60)
    
    # Noise precision
    print(f"\nNoise precision (log_tau):")
    print(f"  value = {model.log_noise_precision.item():.6f}")
    print(f"  tau = exp(value) = {torch.exp(model.log_noise_precision).item():.6f}")
    
    # Layer 1
    print(f"\nLayer 0 (1 -> 20):")
    print(f"  Weight mu shape: {model.layer1.weight_mu.shape}")
    print(f"  Weights:")
    weight_sigma_1 = torch.exp(0.5 * model.layer1.weight_logvar)
    for i in range(model.layer1.weight_mu.shape[0]):
        for j in range(model.layer1.weight_mu.shape[1]):
            print(f"    W[{i},{j}]: mu={model.layer1.weight_mu[i,j].item():.6f}, "
                  f"logvar={model.layer1.weight_logvar[i,j].item():.6f}, "
                  f"sigma={weight_sigma_1[i,j].item():.6f}")
    
    print(f"  Biases:")
    bias_sigma_1 = torch.exp(0.5 * model.layer1.bias_logvar)
    for i in range(model.layer1.bias_mu.shape[0]):
        print(f"    b[{i}]: mu={model.layer1.bias_mu[i].item():.6f}, "
              f"logvar={model.layer1.bias_logvar[i].item():.6f}, "
              f"sigma={bias_sigma_1[i].item():.6f}")
    
    # Layer 2
    print(f"\nLayer 1 (20 -> 1):")
    print(f"  Weight mu shape: {model.layer2.weight_mu.shape}")
    print(f"  Weights:")
    weight_sigma_2 = torch.exp(0.5 * model.layer2.weight_logvar)
    for i in range(model.layer2.weight_mu.shape[0]):
        for j in range(model.layer2.weight_mu.shape[1]):
            print(f"    W[{i},{j}]: mu={model.layer2.weight_mu[i,j].item():.6f}, "
                  f"logvar={model.layer2.weight_logvar[i,j].item():.6f}, "
                  f"sigma={weight_sigma_2[i,j].item():.6f}")
    
    print(f"  Biases:")
    bias_sigma_2 = torch.exp(0.5 * model.layer2.bias_logvar)
    for i in range(model.layer2.bias_mu.shape[0]):
        print(f"    b[{i}]: mu={model.layer2.bias_mu[i].item():.6f}, "
              f"logvar={model.layer2.bias_logvar[i].item():.6f}, "
              f"sigma={bias_sigma_2[i].item():.6f}")
    
    print("\n" + "="*60)
    
    # 5. Plot results
    print("\n[5] Creating visualization...")
    plot_results(X_train, y_train, X_test, y_test, mean_pred, std_pred, losses)
    
    print("\n" + "="*60)
    print("Done! The BNN successfully learned the polynomial curve with uncertainty.")
    print("="*60)


if __name__ == "__main__":
    main()
