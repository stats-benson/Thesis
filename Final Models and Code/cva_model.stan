// ─────────────────────────────────────────────────────────────────────────────
// Contextualized Value-Added Model (CVA)
//
// y_it = mu0 + beta * y_{i,t-1} + gamma' * X_it + theta_{j(i)} + eps_it
//
// theta_j ~ Normal(0, tau^2)   [teacher random effects]
// eps_it  ~ Normal(0, sigma^2) [residual error]
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;                        // number of student-time observations
  int<lower=1> J;                        // number of teachers
  int<lower=1> K;                        // number of covariates in X
  vector[N] y;                           // outcome scores (standardised)
  vector[N] y_prev;                      // prior achievement scores (standardised)
  matrix[N, K] X;                        // student-level covariate matrix
  array[N] int<lower=1, upper=J> teacher_id;  // teacher index for each obs
}

parameters {
  real mu0;                // intercept
  real beta;               // prior achievement coefficient
  vector[K] gamma;         // covariate coefficients
  vector[J] theta;         // teacher random effects
  real<lower=0> tau;       // teacher effect SD (between-teacher)
  real<lower=0> sigma;     // residual SD (within-teacher)
}

model {
  // ── Priors ──────────────────────────────────────────────────────────────────
  mu0   ~ normal(0, 10);
  beta  ~ normal(0, 1);
  gamma ~ normal(0, 1);
  tau   ~ exponential(1);
  sigma ~ exponential(1);

  // ── Teacher random effects ───────────────────────────────────────────────────
  theta ~ normal(0, tau);

  // ── Likelihood ───────────────────────────────────────────────────────────────
  {
    vector[N] mu;
    for (n in 1:N)
      mu[n] = mu0
              + beta * y_prev[n]
              + dot_product(gamma, X[n])
              + theta[teacher_id[n]];
    y ~ normal(mu, sigma);
  }
}

generated quantities {
  // Posterior predictive draws (for ppc_dens_overlay in bayesplot)
  vector[N] y_rep;
  // Log-likelihood (for LOO-CV via loo package)
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = mu0
                + beta * y_prev[n]
                + dot_product(gamma, X[n])
                + theta[teacher_id[n]];
    y_rep[n]   = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
  }
}
