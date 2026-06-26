import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# Example 2 (corrected):
#
# x(t) = 1 + (1/5000) ∫_0^t (t - 2s) x(s) Δs
#
# Time scale T = {0,1,2,4,8,16,32}
# ==========================================================

# ----------------------------------------------------------
# Time scale and graininess
# ----------------------------------------------------------

t_grid = np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.float32)
mu = np.array([1, 1, 2, 4, 8, 16, 32], dtype=np.float32)   # sigma(t)=2t, sigma(0)=1

N = len(t_grid) - 1

t_tensor = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
mu_torch = torch.tensor(mu, dtype=torch.float32)

# ----------------------------------------------------------
# Exact solution using corrected recursion
# ----------------------------------------------------------

def exact_solution_corrected():
    x = np.zeros(len(t_grid))
    x[0] = 1.0
    for n in range(1, len(t_grid)):
        s = 0.0
        for k in range(n):
            s += mu[k] * (t_grid[n] - 2.0 * t_grid[k]) * x[k]
        x[n] = 1.0 + (1.0 / 5000.0) * s
    return x

x_exact = exact_solution_corrected()
print("\nExact solution (corrected):", x_exact)

# ----------------------------------------------------------
# Neural Network (increased capacity)
# ----------------------------------------------------------

class NeuralVolterra(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, t):
        # Enforce x(0)=1 exactly: x(t) = 1 + t * N(t)
        return 1.0 + t * self.net(t)

model = NeuralVolterra()

# ----------------------------------------------------------
# Discrete operator (corrected)
# ----------------------------------------------------------

def Phi(X):
    """
    X: tensor of shape (N+1,) containing values at grid points
    Returns: tensor of shape (N+1,) with Φ(X)_n
    """
    values = []
    for n in range(N + 1):
        if n == 0:
            integral = torch.tensor(0.0, dtype=torch.float32)
        else:
            integral = torch.sum(
                mu_torch[:n] * (t_tensor[n] - 2.0 * t_tensor[:n].squeeze()) * X[:n]
            )
        values.append(1.0 + (1.0 / 5000.0) * integral)
    return torch.stack(values)

# ----------------------------------------------------------
# Optimizer and Scheduler
# ----------------------------------------------------------

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

epochs = 20000
loss_history = []
error_history = []
epoch_history = []

best_error = np.inf
counter = 0
patience = 2000   # early stopping based on error

for epoch in range(epochs):
    optimizer.zero_grad()

    X_theta = model(t_tensor).squeeze()
    Phi_X = Phi(X_theta)
    residual = X_theta - Phi_X
    loss = torch.mean(residual**2)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()   # step every epoch

    # Record history
    with torch.no_grad():
        pred = model(t_tensor).squeeze().cpu().numpy()
        err = np.max(np.abs(pred - x_exact))

    epoch_history.append(epoch)
    loss_history.append(loss.item())
    error_history.append(err)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.2e} | Error = {err:.2e}")

    # Early stopping based on error improvement
    if err < best_error:
        best_error = err
        counter = 0
    else:
        counter += 1

    if counter > patience:
        print(f"\nEarly stopping at epoch {epoch} (error not improved for {patience} epochs)")
        break

# ----------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------

with torch.no_grad():
    X_pred = model(t_tensor).squeeze().cpu().numpy()

abs_error = np.abs(X_pred - x_exact)
E_inf = np.max(abs_error)
E2 = np.sqrt(np.mean(abs_error**2))
final_loss = loss_history[-1]

print("\n" + "="*60)
print("FINAL RESULTS (corrected formula, improved training)")
print("="*60)
print(f"Final loss       = {final_loss:.6e}")
print(f"E_inf (max error)= {E_inf:.6e}")
print(f"E2 (RMSE)        = {E2:.6e}")
print("="*60)

# ----------------------------------------------------------
# Tables
# ----------------------------------------------------------

# Solution table
solution_df = pd.DataFrame({
    "t_i": t_grid,
    "Exact solution": x_exact,
    "Neural solution": X_pred,
    "Absolute error": abs_error
})
print("\nSolution Table\n", solution_df)

# Convergence table (selected epochs)
selected_epochs = [0, 50, 100, 500, 1000, 1500, 2000, 3000, 5000]
# Add final epoch only if not already present
if epoch_history[-1] not in selected_epochs:
    selected_epochs.append(epoch_history[-1])
selected_epochs = sorted(set(selected_epochs))

rows = []
for ep in selected_epochs:
    idx = np.argmin(np.abs(np.array(epoch_history) - ep))
    rows.append([epoch_history[idx], loss_history[idx], error_history[idx]])
conv_df = pd.DataFrame(rows, columns=["Epoch", "Loss", "Error"])
print("\nConvergence Table\n", conv_df)

# Save tables to CSV
solution_df.to_csv("example2_solution_corrected_improved.csv", index=False)
conv_df.to_csv("example2_convergence_corrected_improved.csv", index=False)
print("\nTables saved to CSV files.")

# ==========================================================
# PLOTTING – PURE TIME SCALE WITH CLEARLY DISTINGUISHABLE MARKERS
# ==========================================================

# 1. Exact vs Neural – scatter with distinct markers (no lines)
plt.figure(figsize=(8, 5))
plt.scatter(t_grid, x_exact, color='blue', s=120, marker='o', label='Exact')
plt.scatter(t_grid, X_pred, color='red', s=80, marker='s', label='Neural')
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.title('Exact vs Neural Solution')
plt.legend()
plt.grid(True)
plt.savefig("example2_solution_improved.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Loss convergence – lines (training history, not time scale)
plt.figure(figsize=(8, 5))
plt.semilogy(epoch_history, loss_history, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.grid(True)
plt.savefig("example2_loss_improved.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Error convergence – lines (training history)
plt.figure(figsize=(8, 5))
plt.semilogy(epoch_history, error_history, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Maximum Error')
plt.title('Error History')
plt.grid(True)
plt.savefig("example2_error_improved.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Pointwise absolute error – stem plot
plt.figure(figsize=(8, 5))
plt.stem(t_grid, abs_error, basefmt=" ", linefmt='r-', markerfmt='ro')
plt.xlabel(r'$t$')
plt.ylabel('Absolute Error')
plt.title('Pointwise Error')
plt.grid(True)
plt.savefig("example2_pointwise_error_improved.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Combined figure – loss and error histories (both training curves)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(epoch_history, loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.title('Loss history')
plt.subplot(1, 2, 2)
plt.semilogy(epoch_history, error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.title('Error history')
plt.tight_layout()
plt.savefig("example2_combined_improved.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nAll plots saved as PNG files.")