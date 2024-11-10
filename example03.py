import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network for f(y_p)
class FModel(nn.Module):
    def __init__(self):
        super(FModel, self).__init__()
        self.hidden = nn.Linear(1, 10)  # hidden layer
        self.output = nn.Linear(10, 1)  # output layer
        
    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # sine activation for hidden layer
        x = self.output(x)
        return x

# Define the neural network for g(u)
class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

# Instantiate the models
f_model = FModel()
g_model = GModel()

# Define optimizers and loss function
# Define a single optimizer for both models
optimizer = optim.SGD(list(f_model.parameters()) + list(g_model.parameters()), lr=0.01)
loss_fn = nn.MSELoss()

# Simulation parameters
num_train_steps = 100000  # Training phase
y_p = [0, 0]  # plant output values
errors = []

# Generate random input within [-2, 2] for g_model and [-10, 10] for f_model during training
u_g = [np.random.uniform(-2, 2) for _ in range(num_train_steps)]

# Training loop with combined output and error
for k in range(2, num_train_steps):
    # Current inputs
    u_k = u_g[k]
    y_k = y_p[-1]  # last plant output

    # Compute plant's actual next output using the nonlinear functions
    f_y = y_k / (1 + y_k**2)  # f(y_p(k))
    g_u = u_k**3  # g(u(k))
    y_p_k1 = f_y + g_u  # plant's next output
    y_p.append(y_p_k1)  # Append to plant output history

    # Forward pass through both models with appropriate inputs
    y_tensor = torch.tensor([[y_k]], dtype=torch.float32)  # input to f(y)
    u_tensor = torch.tensor([[u_k]], dtype=torch.float32)  # input to g(u)
    f_hat_y = f_model(y_tensor)  # Model's approximation of f(y_p(k))
    g_hat_u = g_model(u_tensor)  # Model's approximation of g(u(k))
    j_p_k1 = f_hat_y + g_hat_u  # Combined model output for y_p(k+1)

    # Calculate combined error as the difference between plant and model output
    target = torch.tensor([[y_p_k1]], dtype=torch.float32)  # plant's actual output as target
    total_loss = loss_fn(j_p_k1, target)  # Compute combined error

    # Backpropagate the combined error through both networks
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Log the total loss
    errors.append(total_loss.item())

    if k % 10000 == 0:
        print(f"Step {k}: Total Loss = {total_loss.item()}")

# After training, switch input to sinusoidal for evaluation
num_eval_steps = 500  # Evaluation phase
u = [np.sin(2 * np.pi * k / 25) + np.sin(2 * np.pi * k / 10) for k in range(num_eval_steps)]

# Evaluation loop
y_eval = [0, 0]  # plant output in evaluation
j_eval = [0, 0]  # model output in evaluation

for k in range(2, num_eval_steps):
    u_k = u[k]
    
    # Compute plant's actual outputs for evaluation
    f_y = y_eval[-1] / (1 + y_eval[-1]**2)
    g_u = u_k**3
    y_p_k1 = f_y + g_u
    y_eval.append(y_p_k1)
    
    # Model's output for evaluation
    y_tensor = torch.tensor([[y_eval[-1]]], dtype=torch.float32)
    u_tensor = torch.tensor([[u_k]], dtype=torch.float32)
    f_hat_y = f_model(y_tensor)
    g_hat_u = g_model(u_tensor)
    j_p_k1 = f_hat_y.item() + g_hat_u.item()
    j_eval.append(j_p_k1)

# Plot plant output vs model output during evaluation
plt.figure(figsize=(12, 6))

# Plant output during evaluation
plt.subplot(2, 1, 1)
plt.plot(y_eval, label="Plant Output", color='green')
plt.title("Plant's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

# Model output during evaluation
plt.subplot(2, 1, 2)
plt.plot(j_eval, label="Model Output", color='red')
plt.title("Model's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

plt.tight_layout()
plt.show()
