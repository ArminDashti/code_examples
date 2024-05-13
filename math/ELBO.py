import numpy as np
from scipy.stats import norm

def elbo(x, mu, sigma, q_mu, q_sigma):
    samples = np.random.normal(q_mu, q_sigma, 1000)
    expected_log_likelihood = np.mean(np.log(norm.pdf(x, loc=samples)))
    kl_divergence = np.log(q_sigma/sigma) + (sigma**2 + (mu - q_mu)**2) / (2 * q_sigma**2) - 0.5
    return expected_log_likelihood - kl_divergence


x_observed = 1.0  # Observed data point
p_mu = 0.0        # Mean of p(Z)
p_sigma = 1.0     # Std of p(Z)
q_mu = 0.5        # Mean of q(Z)
q_sigma = 0.8     # Std of q(Z)

elbo_value = elbo(x_observed, p_mu, p_sigma, q_mu, q_sigma)
print(f"The ELBO value is: {elbo_value}")
