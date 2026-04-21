// ─────────────────────────────────────────────────────────────────────────────
// Signal-Weighted Value-Added Model (SW VAM) — One-Step Version
// Kim, E.J. (2022). Signal Weighted Teacher Value-Added Models.
// Statistics and Public Policy, 9(1), 149-162.
//
// Kim notes a one-step version is "easily implementable" (Section 2).
// This version takes y directly as the outcome (same as CVA), making
// the two models directly comparable via LOO-CV.
//
// MODEL:
//   //   y_ijt ~ Normal(mu_t + alpha_i * theta_j + beta * y_{i,t-1} + gamma' * X_ijt, 1/sqrt(a_i))
//
// where:
//   mu_t    = time fixed effect (one per wave)
//   alpha_i = student sensitivity  (lognormal, centred at 0.2)
//   theta_j = teacher effect       (normal, sd = 1)
//   gamma   = covariate coefficients (same X as CVA)
//   a_i     = student reliability / precision (lognormal, centred at 1)
//             variance = 1/a_i, so SD = 1/sqrt(a_i)
//   beta    = prior achievement coefficient
//
// PRIORS (Kim Equation 5, lognormal):
//   theta_j  ~ Normal(0, 1)
//   alpha_i  ~ LogNormal(log(0.2), log(2)/2)
//   a_i      ~ LogNormal(log(1),   log(2)/2)
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;                               // observations
  int<lower=1> I;                               // students
  int<lower=1> J;                               // teachers
  int<lower=1> K;                               // covariates
  int<lower=1> TT;                              // time points (waves)
  vector[N] y;                                  // standardised outcome scores
  matrix[N, K] X;                               // covariate matrix (same as CVA)
  array[N] int<lower=1, upper=I> student_id;
  array[N] int<lower=1, upper=J> teacher_id;
  array[N] int<lower=1, upper=TT> time_id;
  vector[N] y_prev;
}

parameters {
  vector[TT] mu;                // time fixed effects
  vector[K] gamma;              // covariate coefficients
  vector[J] theta;              // teacher effects
  vector<lower=0>[I] alpha;     // student sensitivity
  vector<lower=0>[I] a;         // student reliability (precision)
  real beta;
}

model {
  // ── Priors ───────────────────────────────────────────────────────────────────
  mu     ~ normal(0, 1);
  gamma  ~ normal(0, 1);
  theta  ~ normal(0, 1);
  alpha  ~ lognormal(log(0.2), log(2) / 2.0);
  a      ~ lognormal(log(1.0), log(2) / 2.0);
  beta ~ normal(0, 1);

  // ── Likelihood ───────────────────────────────────────────────────────────────
  for (n in 1:N) {
    real mu_n = mu[time_id[n]]
            + alpha[student_id[n]] * theta[teacher_id[n]]
            + beta * y_prev[n]
            + dot_product(gamma, X[n]);
    real sigma_n = 1.0 / sqrt(a[student_id[n]]);
    y[n] ~ normal(mu_n, sigma_n);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n    = mu[time_id[n]]
               + alpha[student_id[n]] * theta[teacher_id[n]]
               + beta * y_prev[n]
               + dot_product(gamma, X[n]);
    real sigma_n = 1.0 / sqrt(a[student_id[n]]);
    y_rep[n]   = normal_rng(mu_n, sigma_n);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
