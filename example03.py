import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# define NN for Nf
class FModel(nn.Module):
    def __init__(self):
        super(FModel, self).__init__()
        self.hidden = nn.Linear(1, 10)  # hidden layer
        self.output = nn.Linear(10, 1)  # output layer
        
    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # apply the tanh activation function
        x = self.output(x)
        return x

# define NN for Ng
class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.tanh(self.hidden(x)) # apply the tanh activation function
        x = self.output(x)
        return x

f_model = FModel()
g_model = GModel()

# define optimizer (for both models) and loss function
optimizer = optim.SGD(list(f_model.parameters()) + list(g_model.parameters()), lr=0.01)
loss_fn = nn.MSELoss()

num_train_steps = 100000
y_p = [0, 0]  # y hat inital output value
errors = []

# generate random input within [-2, 2] for GModel
u_g = [np.random.uniform(-2, 2) for _ in range(num_train_steps)]

# training loop
for k in range(2, num_train_steps):
    # define inputs
    u_k = u_g[k]
    y_k = y_p[-1]  # y hat previous step

    # compute y hat next output using the nonlinear functions
    f_y = y_k / (1 + y_k**2)  # f(y_p(k))
    g_u = u_k**3  # g(u(k))
    y_p_k1 = f_y + g_u  # y hat output
    y_p.append(y_p_k1)  # append to y hat output history

    # forward pass 
    y_tensor = torch.tensor([[y_k]], dtype=torch.float32)  # convert input of f(y) to tensor
    u_tensor = torch.tensor([[u_k]], dtype=torch.float32)  # convert input of g(u) to tensor
    f_hat_y = f_model(y_tensor)
    g_hat_u = g_model(u_tensor)
    j_p_k1 = f_hat_y + g_hat_u  # combined model output for y_p(k+1)

    # calculate combined error as the difference between y hat and model's estimated output
    target = torch.tensor([[y_p_k1]], dtype=torch.float32) 
    total_loss = loss_fn(j_p_k1, target)

    # backpropagate
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    errors.append(total_loss.item())

    if k % 10000 == 0:
        print(f"Step {k}: Total Loss = {total_loss.item()}")

# evaluation loop
num_eval_steps = 500
u = [np.sin(2 * np.pi * k / 25) + np.sin(2 * np.pi * k / 10) for k in range(num_eval_steps)]

y_eval = [0, 0]  # plant inital output
j_eval = [0, 0]  # y hat inital output

for k in range(2, num_eval_steps):
    u_k = u[k]
    
    # compute plant's output
    f_y = y_eval[-1] / (1 + y_eval[-1]**2)
    g_u = u_k**3
    y_p_k1 = f_y + g_u
    y_eval.append(y_p_k1)
    
    # compute y hat's output
    y_tensor = torch.tensor([[y_eval[-1]]], dtype=torch.float32)
    u_tensor = torch.tensor([[u_k]], dtype=torch.float32)
    f_hat_y = f_model(y_tensor)
    g_hat_u = g_model(u_tensor)
    j_p_k1 = f_hat_y.item() + g_hat_u.item()
    j_eval.append(j_p_k1)

# plot plant output vs model
plt.figure(figsize=(12, 6))

# plant output
plt.subplot(2, 1, 1)
plt.plot(y_eval, label="Plant Output", color='green')
plt.title("Plant's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

# model output
plt.subplot(2, 1, 2)
plt.plot(j_eval, label="Model Output", color='red')
plt.title("Model's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

plt.tight_layout()
plt.show()
