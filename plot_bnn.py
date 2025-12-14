#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
pred_data = pd.read_csv('bnn_predictions.csv')
train_data = pd.read_csv('bnn_training.csv')
loss_data = pd.read_csv('bnn_loss.csv')

# ===== Compute Loss from Predictions =====
print("="*60)
print("BNN Performance Metrics (Computed from Data)")
print("="*60)

# Compute RMSE on test data (predictions)
mse_test = np.mean((pred_data['mean'] - pred_data['true_y'])**2)
rmse_test = np.sqrt(mse_test)
print(f"Test RMSE: {rmse_test:.6f}")
print(f"Test MSE:  {mse_test:.6f}")

# Compute RMSE on training data
# Need to get predictions for training points - approximate using nearest x values
train_errors = []
for idx, row in train_data.iterrows():
    x_train = row['x']
    y_train = row['y']
    # Find closest x in predictions
    closest_idx = np.argmin(np.abs(pred_data['x'] - x_train))
    y_pred = pred_data['mean'].iloc[closest_idx]
    train_errors.append((y_pred - y_train)**2)

mse_train = np.mean(train_errors)
rmse_train = np.sqrt(mse_train)
print(f"Training RMSE (approx): {rmse_train:.6f}")
print(f"Training MSE (approx):  {mse_train:.6f}")

# Mean prediction uncertainty
mean_uncertainty = pred_data['std'].mean()
print(f"\nMean prediction uncertainty (std): {mean_uncertainty:.6f}")

# Coverage statistics (what % of true values fall within confidence intervals)
within_bounds = ((pred_data['true_y'] >= pred_data['lower']) & 
                 (pred_data['true_y'] <= pred_data['upper'])).sum()
coverage = (within_bounds / len(pred_data)) * 100
print(f"95% Confidence interval coverage: {coverage:.1f}%")

print("\n" + "="*60)
print("Training Loss History")
print("="*60)

# ===== Compute and Print Loss Statistics =====
initial_loss = loss_data['loss_per_point'].iloc[0]
final_loss = loss_data['loss_per_point'].iloc[-1]
min_loss = loss_data['loss_per_point'].min()
mean_loss_last_1000 = loss_data['loss_per_point'].iloc[-1000:].mean() if len(loss_data) >= 1000 else loss_data['loss_per_point'].mean()
improvement = ((initial_loss - final_loss) / initial_loss) * 100

print(f"Initial loss (per point): {initial_loss:.6f}")
print(f"Final loss (per point):   {final_loss:.6f}")
print(f"Minimum loss (per point): {min_loss:.6f}")
print(f"Mean loss (last 1000 epochs): {mean_loss_last_1000:.6f}")
print(f"Improvement: {improvement:.2f}%")
print(f"Total epochs: {len(loss_data)}")
print("="*60)
print()

# ===== Plot 1: Predictions and Uncertainty =====
fig, ax = plt.subplots(figsize=(10, 6))

# Plot true function
ax.plot(pred_data['x'], pred_data['true_y'], 'g-', label='True -2x^2 + 10x + 5', linewidth=2)

# Plot mean prediction
ax.plot(pred_data['x'], pred_data['mean'], 'b-', label='BNN Mean Prediction', linewidth=2)

# Plot uncertainty bands (2 std)
ax.fill_between(pred_data['x'], pred_data['lower'], pred_data['upper'],
                 alpha=0.3, color='blue', label='95% Confidence')

# Plot training data
ax.scatter(train_data['x'], train_data['y'], c='red', s=20, alpha=0.5, label='Training Data')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Bayesian Neural Network - Sin Function Approximation', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('bnn_results_polynomial_lowerbeta.png', dpi=300, bbox_inches='tight')
print('Prediction plot saved to bnn_results.png')
plt.close()

# ===== Plot 2: Training Loss Over Time =====
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(loss_data['epoch'], loss_data['loss_per_point'], 'b-', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss per Data Point', fontsize=12)
ax2.set_title('BNN Training Loss (ELBO per point)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add text with final loss
final_loss = loss_data['loss_per_point'].iloc[-1]
initial_loss = loss_data['loss_per_point'].iloc[0]
improvement = ((initial_loss - final_loss) / initial_loss) * 100
ax2.text(0.05, 0.95, f'Initial: {initial_loss:.4f}\nFinal: {final_loss:.4f}\nImprovement: {improvement:.1f}%',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('bnn_loss_polynomial_lowerbeta.png', dpi=300, bbox_inches='tight')
print('Loss plot saved to bnn_loss.png')
plt.close()

print('All plots saved successfully!')
