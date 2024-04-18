import torch
from torch import nn
from torch.distributions import MultivariateNormal

# Define a simple model (true distribution)
class TrueModel(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(TrueModel, self).__init__()
    self.linear = nn.Linear(dim_in, dim_out)

  def forward(self, x):
    return self.linear(x)

# Sample some data points from the true model
dim_in = 2  # Input dimension
dim_out = 1  # Output dimension
num_data = 1000
true_model = TrueModel(dim_in, dim_out)
data = true_model(torch.randn(num_data, dim_in))

# Define the variational distribution (approximate posterior)
class VariationalDistribution(nn.Module):
  def __init__(self, dim_in, dim_latent):
    super(VariationalDistribution, self).__init__()
    self.fc1 = nn.Linear(dim_in, dim_latent)
    self.fc2_mu = nn.Linear(dim_latent, dim_latent)
    self.fc2_logvar = nn.Linear(dim_latent, dim_latent)

  def forward(self, x):
    hidden = torch.relu(self.fc1(x))
    mu = self.fc2_mu(hidden)
    logvar = self.fc2_logvar(hidden)
    std = torch.exp(0.5 * logvar)  # Ensure positive standard deviation
    return MultivariateNormal(mu, torch.diag(std))

# Define the variational inference (VI) function 
def vi_step(data, variational_distribution):
  # Get the approximate posterior distribution
  q_z = variational_distribution(data)

  # Define the KL divergence between approximate posterior and prior (usually standard normal)
  # (Here, we omit the constant term for simplicity)
  kl_divergence = -0.5 * torch.sum(q_z.log_prob(q_z.sample()) - MultivariateNormal(torch.zeros(dim_latent), torch.eye(dim_latent)).log_prob(q_z.sample()))

  # Define the reconstruction loss (e.g., mean squared error) between data and reconstructed data from q_z
  # (This depends on your true model and data)
  reconstruction_loss = torch.nn.functional.mse_loss(true_model(data), data)

  # Combine losses (ELBO objective)
  elbo = -kl_divergence - reconstruction_loss
  return elbo

# Define variational distribution with a latent dimension of 3 
variational_distribution = VariationalDistribution(dim_in, 3)

# Train the variational distribution by maximizing the ELBO 
optimizer = torch.optim.Adam(variational_distribution.parameters())
for epoch in range(100):
  optimizer.zero_grad()
  elbo = vi_step(data, variational_distribution)
  elbo.backward()
  optimizer.step()

# After training, the variational distribution `q_z` approximates the posterior distribution 
# of the data points with respect to the true model. 
