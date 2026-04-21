// ─────────────────────────────────────────────────────────────────────────────
// Unified Value-Added Model
// Combines:
//   - Signal weighting (Kim 2022 SWVAM): alpha_i * theta_j, 1/a_i precision
//   - Covariate bias reduction (CVA/Composite): demographics, peer, school
//   - Longitudinal covariance (TVAAS): unrestricted Sigma across waves
//
// For student i observed at waves idx[1..k]:
//   Y_i ~ MVN(mu_i, Sigma_i)
//
// where:
//   mu_i[p] = nu[g]                     [wave mean]
//           + mu0                        [intercept]
//           + alpha_i * theta_j          [signal-weighted teacher effect]
//           + gamma' * X_i              [student demographics]
//           + delta * P_ig              [peer achievement]
//           + zeta' * S_j               [school characteristics]
//
//   Sigma_i = Sigma / a_i               [student-scaled cross-wave covariance]
//
// IDENTIFICATION:
//   - theta_j non-centred: theta = tau_theta * theta_raw
//   - alpha_i * theta_j scale anchored by tau_theta ~ Exponential(1)
//   - a_i scales overall precision, Sigma captures correlation structure
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;    // students
  int<lower=1> J;    // teachers
  int<lower=1> G;    // waves
  int<lower=1> K;    // student covariates
  int<lower=1> L;    // school covariates

  matrix[N, G] Y;                               // wide score matrix
  matrix[N, K] X;                               // student demographics (time-invariant)
  matrix[N, G] P;                               // peer achievement per wave
  matrix[J, L] S;                               // school covariates per teacher

  array[N, G] int<lower=0, upper=J> teacher_id; // 0 = unobserved
  array[N, G] int<lower=0, upper=1> obs;         // observation mask
  array[N]    int<lower=1, upper=G> n_obs;       // observed waves per student
}

parameters {
  // ── Fixed effects ──────────────────────────────────────────────────────────
  vector[G] nu;           // wave means
  real mu0;               // intercept
  real delta;             // peer achievement effect
  vector[K] gamma;        // student demographic coefficients
  vector[L] zeta;         // school characteristic coefficients

  // ── Teacher effects (non-centred) ─────────────────────────────────────────
  vector[J] theta_raw;
  real<lower=0> tau_theta;

  // ── Student parameters (Kim 2022) ─────────────────────────────────────────
  vector<lower=0>[N] alpha;   // sensitivity: scales teacher effect in mean
  vector<lower=0>[N] a;       // reliability: scales overall precision

  // ── Cross-wave covariance (TVAAS) ─────────────────────────────────────────
  cholesky_factor_corr[G] L_corr;
  vector<lower=0>[G] sigma_wave;
}

transformed parameters {
  // Non-centred teacher effects
  vector[J] theta = tau_theta * theta_raw;

  // Full cross-wave covariance matrix
  matrix[G, G] Sigma = diag_pre_multiply(sigma_wave, L_corr) *
                       diag_pre_multiply(sigma_wave, L_corr)';
}

model {
  // ── Priors ─────────────────────────────────────────────────────────────────
  nu         ~ normal(0, 1);
  mu0        ~ normal(0, 10);
  delta      ~ normal(0, 1);
  gamma      ~ normal(0, 1);
  zeta       ~ normal(0, 1);
  theta_raw  ~ normal(0, 1);
  tau_theta  ~ exponential(1);
  alpha      ~ lognormal(log(0.2), log(2) / 2.0);
  a          ~ lognormal(log(1.0), log(2) / 2.0);
  L_corr     ~ lkj_corr_cholesky(2);
  sigma_wave ~ exponential(1);

  // ── Likelihood ─────────────────────────────────────────────────────────────
  for (i in 1:N) {
    int k = n_obs[i];
    array[k] int idx;
    int pos = 1;

    // Collect observed wave indices
    for (g in 1:G) {
      if (obs[i, g] == 1) {
        idx[pos] = g;
        pos += 1;
      }
    }

    // Build mean vector across observed waves
    vector[k] mu_i;
    for (p in 1:k) {
      int g = idx[p];
      int j = teacher_id[i, g];
      mu_i[p] = nu[g]
                + mu0
                + dot_product(gamma, X[i]')
                + delta * P[i, g];
      if (j > 0) {
        mu_i[p] += alpha[i] * theta[j]
                +  dot_product(zeta, S[j]');
      }
    }

    // Student-scaled submatrix of Sigma
    // a_i scales overall precision: Sigma_i = Sigma / a_i
    matrix[k, k] Sigma_i;
    for (p in 1:k) {
      for (q in 1:k) {
        Sigma_i[p, q] = Sigma[idx[p], idx[q]] / a[i];
      }
    }

    // Observed scores
    vector[k] y_i;
    for (p in 1:k) {
      y_i[p] = Y[i, idx[p]];
    }

    y_i ~ multi_normal(mu_i, Sigma_i);
  }
}

generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    int k = n_obs[i];
    array[k] int idx;
    int pos = 1;

    for (g in 1:G) {
      if (obs[i, g] == 1) {
        idx[pos] = g;
        pos += 1;
      }
    }

    vector[k] mu_i;
    for (p in 1:k) {
      int g = idx[p];
      int j = teacher_id[i, g];
      mu_i[p] = nu[g]
                + mu0
                + dot_product(gamma, X[i]')
                + delta * P[i, g];
      if (j > 0) {
        mu_i[p] += alpha[i] * theta[j]
                +  dot_product(zeta, S[j]');
      }
    }

    matrix[k, k] Sigma_i;
    for (p in 1:k) {
      for (q in 1:k) {
        Sigma_i[p, q] = Sigma[idx[p], idx[q]] / a[i];
      }
    }

    vector[k] y_i;
    for (p in 1:k) {
      y_i[p] = Y[i, idx[p]];
    }

    log_lik[i] = multi_normal_lpdf(y_i | mu_i, Sigma_i);
  }
}