import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# define the structure of the NN
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = nn.Linear(1, 10)  # hidden layer
        self.output = nn.Linear(10, 1)  # output layer
        
    def forward(self, x):
        x = torch.sin(self.hidden(x))  # apply the sine activation function
        x = self.output(x)
        return x

model = Model()

# define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.25)
loss_fn = nn.MSELoss()

num_steps = 5000  # reduced for efficiency and ease of visualization

# define inital output values
y_p = [0, 0]  # plant
j_p = [0, 0]  # model
errors = []

# generate inputs. sinusoidal input signal for first 500 steps, then randomly uniformly distributes [-1,1]
u = [np.sin(2 * np.pi * k / 250) for k in range(500)]
u += [np.random.uniform(-1, 1) for _ in range(num_steps - 500)]

# training loop
for k in range(2, num_steps):
    # compute plant's output
    u_k = u[k]
    f_u = 0.6 * np.sin(np.pi * u_k) + 0.3 * np.sin(3 * np.pi * u_k) + 0.1 * np.sin(5 * np.pi * u_k) # f(u) = 0.6 sin(nu) + 0.3 sin (37ru) + 0.1 sin (5nu)
    y_p_k1 = 0.3 * y_p[-1] + 0.6 * y_p[-2] + f_u # yp(k) = 0.3yp(k - 1) + 0.6yp(k - 2) + f[u(k)]
    y_p.append(y_p_k1) # adds the calculated value of yp(k + 1) to the next output of the plant to reuse in future time stamps
    
    # convert the u input to a tensor
    u_tensor = torch.tensor([[u_k]], dtype=torch.float32)
    
    # forward pass
    f_hat_u = model(u_tensor)
    j_p_k1 = 0.3 * y_p[-1] + 0.6 * y_p[-2] + f_hat_u.item() # jp(k) = 0.3yp(k - 1) + 0.6yp(k - 2) + N[u(k)]
    j_p.append(j_p_k1) # adds the calculated value of jp(k + 1) to the next output of the plant to reuse in future time stamps
    
    # calculate error and backpropagate
    target = torch.tensor([[f_u]], dtype=torch.float32)
    loss = loss_fn(f_hat_u, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    errors.append(loss.item())

    if k % 500 == 0:
        print(f"Step {k}: Loss = {loss.item()}")

# plot plant output vs model output
plt.figure(figsize=(12, 6))

# plant output
plt.subplot(2, 1, 1)
plt.plot(y_p, label="Plant Output", color='green')
plt.title("Plant's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

# model output
plt.subplot(2, 1, 2)
plt.plot(j_p, label="Model Output", color='red')
plt.title("Model's Output")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()

plt.tight_layout()
plt.show()
