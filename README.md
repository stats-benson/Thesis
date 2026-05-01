USE THE FOLLOWING FILE DOWNLOAD = https://nces.ed.gov/datalab/onlinecodebook



https://pmc.ncbi.nlm.nih.gov/articles/PMC2743034/#S34 


signal weighted value added models




Due to ECLS-K's within-classroom sampling design, most teachers appear with only 2–4 sampled students. Following [cite VAM literature], we restrict analysis to teachers with at least 5 observations, retaining X teachers. We note that estimates for all models will reflect substantial shrinkage toward zero under sparse data, and interpret cross-model comparisons accordingly.








# Load your stability fits
fit_cva_A <- readRDS("stan_fits/fit_cva_A.rds")
fit_cva_B <- readRDS("stan_fits/fit_cva_B.rds")

# Get teacher index mapping
# (use your existing get_theta helper)
theta_A <- get_theta(fit_cva_A, teacher_index_A, "theta_A")
theta_B <- get_theta(fit_cva_B, teacher_index_B, "theta_B")

# Join matched teachers
matched <- inner_join(theta_A, theta_B, by = "teacher_id")

# Plot cohort A vs cohort B
ggplot(matched, aes(x = theta_A, y = theta_B)) +
  geom_point(alpha = 0.4, color = "#7F77DD", size = 2) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey40") +
  geom_smooth(method = "lm", se = TRUE,
              color = "#D85A30", fill = "#D85A30", alpha = 0.15) +
  annotate("text", x = -0.4, y = 0.55,
           label = sprintf("r = %.3f", cor(matched$theta_A, matched$theta_B)),
           size = 5, hjust = 0, fontface = "bold", color = "#7F77DD") +
  labs(
    x = "CVA Score — Cohort A",
    y = "CVA Score — Cohort B"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())
