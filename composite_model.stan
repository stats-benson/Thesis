// ─────────────────────────────────────────────────────────────────────────────
// Composite Value-Added Model
// Combines CVA covariate structure with SWVAM student signal parameters
//
// y_ijt = mu_t
//         + a_i * theta_j          [teacher effect, weighted by sensitivity]
//         + r_i * eta_i            [student latent trait, weighted by reliability]
//         + mu0                    [intercept]
//         + beta * y_{i,t-1}       [prior achievement]
//         + gamma' * X_it          [student demographics]
//         + delta * P_it           [peer achievement]
//         + zeta' * S_j            [school characteristics]
//         + eps_ijt
//
// Student signal parameters (from Kim 2022):
//   a_i  ~ LogNormal(log(0.2), log(2)/2)  sensitivity: scales teacher effect
//   r_i  ~ Beta(alpha_r, beta_r)           reliability: scales latent trait
//
// Teacher and student random effects:
//   theta_j ~ Normal(0, 1)         [fixed scale for identification]
//   eta_i   ~ Normal(0, tau_eta)   [student latent trait]
//
// Residual:
//   eps_ijt ~ Normal(0, 1/sqrt(a_i))  [student-specific precision, as in SWVAM]
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;                               // observations
  int<lower=1> I;                               // students
  int<lower=1> J;                               // teachers
  int<lower=1> K;                               // student-level covariates (X)
  int<lower=1> L;                               // school-level covariates (S)
  int<lower=1> TT;                              // time points

  vector[N] y;                                  // standardised outcome
  vector[N] y_prev;                             // standardised prior achievement
  vector[N] P;                                  // peer achievement (mean prior score of school cohort)
  matrix[N, K] X;                               // student covariates
  matrix[J, L] S;                               // school covariates (one row per teacher's school)

  array[N] int<lower=1, upper=I> student_id;
  array[N] int<lower=1, upper=J> teacher_id;
  array[N] int<lower=1, upper=TT> time_id;
}

parameters {
  // ── Fixed effects ────────────────────────────────────────────────────────────
  vector[TT] mu;                // time fixed effects
  real mu0;                     // intercept
  real beta;                    // prior achievement coefficient
  real delta;                   // peer achievement coefficient
  vector[K] gamma;              // student covariate coefficients
  vector[L] zeta;               // school covariate coefficients

  // ── Teacher effects ───────────────────────────────────────────────────────────
  // tau_theta fixed to 1 for identification (same as SWVAM)
  vector[J] theta;

  // ── Student latent trait ──────────────────────────────────────────────────────
  real<lower=0> tau_eta;
  vector[I] eta_raw;            // non-centred: eta = tau_eta * eta_raw

  // ── Student signal parameters ─────────────────────────────────────────────────
  // a_i: sensitivity (lognormal, Kim 2022 Eq 5)
  vector<lower=0>[I] a;

  // r_i: reliability (Beta, as in thesis spec)
  real<lower=0> alpha_r;
  real<lower=0> beta_r;
  vector<lower=0, upper=1>[I] r;
}

transformed parameters {
  vector[I] eta = tau_eta * eta_raw;
}

model {
  // ── Priors ────────────────────────────────────────────────────────────────────
  mu      ~ normal(0, 1);
  mu0     ~ normal(0, 10);
  beta    ~ normal(0, 1);
  delta   ~ normal(0, 1);
  gamma   ~ normal(0, 1);
  zeta    ~ normal(0, 1);

  // Teacher effects (scale fixed to 1 for identification)
  theta   ~ normal(0, 1);

  // Student latent trait
  tau_eta ~ exponential(1);
  eta_raw ~ normal(0, 1);

  // Student sensitivity (Kim 2022 lognormal prior)
  a       ~ lognormal(log(0.2), log(2) / 2.0);

  // Student reliability (Beta, hierarchical)
  alpha_r ~ gamma(2, 0.5);
  beta_r  ~ gamma(2, 0.5);
  r       ~ beta(alpha_r, beta_r);

  // ── Likelihood ────────────────────────────────────────────────────────────────
  for (n in 1:N) {
    int i = student_id[n];
    int j = teacher_id[n];
    int t = time_id[n];

    real mu_n = mu[t]
                + a[i] * theta[j]
                + r[i] * eta[i]
                + mu0
                + beta  * y_prev[n]
                + dot_product(gamma, X[n])
                + delta * P[n]
                + dot_product(zeta, S[j]);

    real sigma_n = 1.0 / sqrt(a[i]);

    y[n] ~ normal(mu_n, sigma_n);
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
                + a[i] * theta[j]
                + r[i] * eta[i]
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
