// Save as sw_vam.stan
data {
  int<lower=1> N; // Total observations
  int<lower=1> S; // Number of students
  int<lower=1> T; // Number of teachers
  int student_id[N];
  int teacher_id[N];
  vector[N] y;
}
parameters {
  vector[T] theta;        // Teacher Value-Added
  vector<lower=0>[S] a;   // Student Reliability
  vector[S] alpha;        // Student Sensitivity
  real<lower=0> sigma;    // Overall noise
}
model {
  // Priors
  theta ~ normal(0, 1);
  a ~ gamma(2, 2);
  alpha ~ normal(1, 0.5);
  
  // SW VAM Likelihood
  for (n in 1:N) {
    y[n] ~ normal(alpha[student_id[n]] * theta[teacher_id[n]], 1 / a[student_id[n]]);
  }
}
