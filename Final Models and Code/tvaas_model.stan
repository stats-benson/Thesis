// ─────────────────────────────────────────────────────────────────────────────
// Tennessee Value-Added Assessment System (TVAAS)
// Sanders & Horn (1994); SAS EVAAS Multivariate Response Model
//
// For a single subject across G waves, each student contributes a
// G-dimensional score vector. The model is:
//
//   Y_i = D_i * nu + sum_t Z_ijt * theta_j + eps_i
//
// where:
//   Y_i       = G-vector of student i's scores across waves
//   D_i       = G x G diagonal indicator matrix (which waves observed)
//   nu        = G-vector of wave-level mean achievement
//   Z_ijt     = 1 if student i assigned to teacher j at wave t
//   theta_j   = teacher effect (scalar random effect, applied per wave)
//   eps_i     ~ MVN(0, Sigma)  with Sigma unrestricted G x G
//
// The unrestricted Sigma is the defining feature of TVAAS. It captures
// the full within-student covariance structure across waves, allowing
// the model to borrow strength across a student's academic history.
//
// IDENTIFICATION:
//   theta_j ~ Normal(0, tau_theta) via non-centred parameterisation.
//   theta_raw ~ Normal(0, 1), theta = tau_theta * theta_raw.
//   This separates the scale (tau_theta) from the direction (theta_raw),
//   avoiding the funnel geometry that causes low E-BFMI with the centred
//   parameterisation when tau_theta is near zero.
//
// MISSING DATA: Students with missing scores at some waves are handled
//   by marginalising over the missing dimensions using the relevant
//   submatrix of Sigma (implemented via indexing in Stan).
//
// NOTE: This model does NOT include demographic covariates. The TVAAS
//   philosophy is that persistent background influences are absorbed by
//   the student's prior score trajectory embedded in Sigma.
// ─────────────────────────────────────────────────────────────────────────────

data {
  int<lower=1> N;               // number of students
  int<lower=1> J;               // number of teachers
  int<lower=1> G;               // number of waves

  matrix[N, G] Y;               // student x wave score matrix (standardised)

  array[N, G] int<lower=0, upper=1> obs;    // observation mask
  array[N] int<lower=1, upper=G>    n_obs;  // observed waves per student

  // teacher_id[i, g] = teacher index for student i at wave g; 0 = unobserved
  array[N, G] int<lower=0, upper=J> teacher_id;
}

parameters {
  vector[G] nu;                         // wave-level mean achievement
  real<lower=0> tau_theta;              // SD of teacher effects
  vector[J] theta_raw;                  // non-centred teacher effects
  cholesky_factor_corr[G] L_corr;      // Cholesky of correlation matrix
  vector<lower=0>[G] sigma_wave;        // per-wave SDs
}

transformed parameters {
  // Non-centred teacher effects: theta = tau_theta * theta_raw
  vector[J] theta = tau_theta * theta_raw;

  // Reconstruct full covariance matrix from Cholesky factors
  matrix[G, G] Sigma = diag_pre_multiply(sigma_wave, L_corr) *
                       diag_pre_multiply(sigma_wave, L_corr)';
}

model {
  // ── Priors ────────────────────────────────────────────────────────────────
  nu          ~ normal(0, 1);
  tau_theta   ~ exponential(1);
  theta_raw   ~ normal(0, 1);           // non-centred — scale handled by tau_theta
  L_corr      ~ lkj_corr_cholesky(2);
  sigma_wave  ~ exponential(1);

  // ── Likelihood ────────────────────────────────────────────────────────────
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
      mu_i[p] = nu[g];
      if (teacher_id[i, g] > 0) {
        mu_i[p] += theta[teacher_id[i, g]];
      }
    }

    matrix[k, k] Sigma_obs;
    for (p in 1:k) {
      for (q in 1:k) {
        Sigma_obs[p, q] = Sigma[idx[p], idx[q]];
      }
    }

    vector[k] y_i;
    for (p in 1:k) {
      y_i[p] = Y[i, idx[p]];
    }

    y_i ~ multi_normal(mu_i, Sigma_obs);
  }
}

generated quantities {
  // Log-likelihood per student (not per observation) for LOO-CV
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
      mu_i[p] = nu[g];
      if (teacher_id[i, g] > 0) {
        mu_i[p] += theta[teacher_id[i, g]];
      }
    }

    matrix[k, k] Sigma_obs;
    for (p in 1:k) {
      for (q in 1:k) {
        Sigma_obs[p, q] = Sigma[idx[p], idx[q]];
      }
    }

    vector[k] y_i;
    for (p in 1:k) {
      y_i[p] = Y[i, idx[p]];
    }

    log_lik[i] = multi_normal_lpdf(y_i | mu_i, Sigma_obs);
  }
}
