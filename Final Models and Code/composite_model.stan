// ─────────────────────────────────────────────────────────────────────────────
// Composite Value-Added Model
// Extends Kim (2022) SWVAM with CVA covariate structure
//
// y_ijt = mu_t
//         + alpha_i * theta_j      [teacher effect scaled by sensitivity]
//         + mu0                    [intercept]
//         + beta * y_{i,t-1}       [prior achievement]
//         + gamma' * X_it          [student demographics]
//         + delta * P_it           [peer achievement]
//         + zeta' * S_j            [school characteristics]
//         + eps_ijt
//
//   eps_ijt ~ Normal(0, 1/sqrt(a_i))   [student-specific precision]
//
// STUDENT PARAMETERS (Kim 2022 Equation 4, two separate parameters):
//   alpha_i = sensitivity  — scales how much teacher quality affects student i
//             alpha_i ~ LogNormal(log(0.2), log(2)/2)   [Kim Equation 5]
//   a_i     = reliability  — precision of student i's signal
//             a_i     ~ LogNormal(log(1.0), log(2)/2)   [Kim Equation 5]
//
// These are SEPARATE parameters, faithful to Kim's IRT interpretation:
//   alpha_i in the mean  (sensitivity)
//   a_i     in variance  (reliability/precision)
//
// IDENTIFICATION:
//   alpha_i * theta_j has a scale ambiguity — multiplying alpha by c and
//   dividing theta by c leaves the mean unchanged. This is anchored by
//   fixing theta_j ~ Normal(0, 1), following Kim's approach.
//
// COMPOSITE CONTRIBUTION beyond Kim (2022):
//   Kim deliberately excluded covariates, applying SWVAM to first-step
//   residuals. This model integrates covariate bias-reduction directly
//   into the signal-weighted likelihood in a single estimation step,
//   addressing both bias (covariates) and variance (signal weighting)
//   simultaneously.
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> L;
  int<lower=1> TT;

  vector[N] y;
  vector[N] y_prev;
  vector[N] P;
  matrix[N, K] X;
  matrix[J, L] S;

  array[N] int<lower=1, upper=I> student_id;
  array[N] int<lower=1, upper=J> teacher_id;
  array[N] int<lower=1, upper=TT> time_id;
}

parameters {
  // ── Fixed effects ─────────────────────────────────────────────────────────
  vector[TT] mu;
  real mu0;
  real beta;
  real delta;
  vector[K] gamma;
  vector[L] zeta;

  // ── Teacher effects (scale fixed to 1 for identification) ─────────────────
  vector[J] theta;

  // ── Student parameters (Kim 2022 Equation 4 — two separate parameters) ───
  vector<lower=0>[I] alpha;   // sensitivity: scales teacher effect in mean
  vector<lower=0>[I] a;       // reliability: controls residual precision
}

model {
  // ── Priors ────────────────────────────────────────────────────────────────
  mu     ~ normal(0, 1);
  mu0    ~ normal(0, 10);
  beta   ~ normal(0, 1);
  delta  ~ normal(0, 1);
  gamma  ~ normal(0, 1);
  zeta   ~ normal(0, 1);

  // Teacher effects (anchors the alpha_i * theta_j scale)
  theta  ~ normal(0, 1);

  // Student sensitivity (Kim Equation 5)
  alpha  ~ lognormal(log(0.2), log(2) / 2.0);

  // Student reliability/precision (Kim Equation 5)
  a      ~ lognormal(log(1.0), log(2) / 2.0);

  // ── Likelihood ────────────────────────────────────────────────────────────
  {
    vector[N] mu_obs;
    vector[N] sigma_obs;
    for (n in 1:N) {
      int i = student_id[n];
      int j = teacher_id[n];
      int t = time_id[n];

      mu_obs[n] = mu[t]
                  + alpha[i] * theta[j]
                  + mu0
                  + beta  * y_prev[n]
                  + dot_product(gamma, X[n])
                  + delta * P[n]
                  + dot_product(zeta, S[j]);

      sigma_obs[n] = 1.0 / sqrt(a[i]);
    }
    y ~ normal(mu_obs, sigma_obs);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    int i = student_id[n];
    int j = teacher_id[n];
    int t = time_id[n];

    real mu_n = mu[t]
                + alpha[i] * theta[j]
                + mu0
                + beta  * y_prev[n]
                + dot_product(gamma, X[n])
                + delta * P[n]
                + dot_product(zeta, S[j]);

    real sigma_n = 1.0 / sqrt(a[i]);

    y_rep[n]   = normal_rng(mu_n, sigma_n);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
