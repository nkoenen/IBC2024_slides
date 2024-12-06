################################################################################
#                   Can We Trust a Neural Network Prediction?                  #
#                 Methods and Pitfalls for Explaining Black Boxes              #
#                                                                              #
#                   Reproduction Materials for IBC 2024 talk                   #
################################################################################
library(data.table)
library(innsight) # installed via devtools::install_github("bips-hb/innsight@activations")
library(ggplot2)
library(ggsci)
library(torch)
library(luz)
library(iml)
library(patchwork)
library(ranger)
library(geomtextpath)
library(Metrics)

# Load utility functions
source("utils.R")

# Set seeds for reproducibility
set.seed(42)
torch_manual_seed(42)

# Load data
data <- read.csv("datasets/diabetes_prediction_dataset.csv")

# Preprocess the data
data <- data[data$gender != "Other", ]
data <- data.table(
  Diabetes = data$diabetes,
  # Binary covariates -----------------------------
  Gender = factor(data$gender, levels = c("Male", "Female"), labels = c(0, 1)),
  # Continuous covariates -------------------------
  BMI = as.numeric(data$bmi),
  Age = as.numeric(data$age),
  HbA1c = data$HbA1c_level
)


# Simulate outcome variable (quadratic effect of BMI)
dat <- data.frame(
  BMI2 =  2* ifelse(data$BMI < 15, 0, pmin((data$BMI - mean(data$BMI))^2, 75)),
  BMI = 0.75 * data$BMI,
  Age = 1 * data$Age,
  HbA1c = 5 * data$HbA1c,
  Gender = 20 * (as.numeric(data$Gender) - 1),
  BMI_Age = 0.5 * pmax(pmin((data$BMI - mean(data$BMI)) * data$Age, 100), -100)
)
out <- rowSums(as.matrix(dat))
data$Diabetes <- as.factor(rbinom(nrow(data), size = 1,
                                  prob = plogis(out - median(out), scale = 10)))

# Create encoded version of the data
data_enc <- data
data_enc$Diabetes <- as.numeric(data$Diabetes) - 1
data_enc$Gender <- as.numeric(data_enc$Gender) - 1


# Split the data into training and testing sets
idx <- sample(nrow(data), nrow(data) * 0.8)
data_train <- data[idx, ]
data_test <- data[-idx, ]
data_train_enc <- data_enc[idx, ]
data_test_enc <- data_enc[-idx, ]


################################################################################
#                       Model-specific methods: Linear Model
################################################################################

# Fit linear model on the data
model_linear <- glm(Diabetes ~ ., data = data_train, family = "binomial")

# Calculate F1 score
pred <- as.numeric(predict(model_linear, data_test, type = "response"))
f1_score <- fbeta_score(as.numeric(data_test$Diabetes) - 1, pred > 0.5)
cat("F1 score (logRegression): ", f1_score, "\n")

# Fit ranger model -------------------------------------------------------------
model_ranger <- ranger(Diabetes ~ ., data = data_train, importance = "impurity")

# Calculate F1 score
pred_test <- predict(model_ranger, data_test)$predictions
f1_score <- fbeta_score(as.numeric(data_test$Diabetes) -1, as.numeric(pred_test) - 1)
cat("F1 score (ranger): ", f1_score, "\n")

# Fit neural network model -----------------------------------------------------

# Create torch datasets
ds_train <- create_dataset(data_train_enc, "Diabetes")
ds_test <- create_dataset(data_test_enc, "Diabetes")


# Fit model
fitted_model <- get_model %>%
  luz::setup(
    loss = nn_bce_loss(),
    metrics = luz_metric_binary_auroc(),
    optimizer = optim_adam
  ) %>%
  set_opt_hparams(lr = 0.005) %>%
  set_hparams(feat_in = ncol(data_train_enc) - 1,
              feat_out = 1,
              classification = TRUE,
              activation = "elu",
              dropout_rate = 0.15) %>%
  fit(ds_train, epochs = 50,
      valid_data = ds_test,
      dataloader_options = list(batch_size = 2048),
      callbacks = list(
        luz_callback_keep_best_model(),
        luz_callback_lr_scheduler(torch::lr_step, step_size = 30, gamma = 0.1)
      ))

model <- fitted_model$model$net
model$eval()

# Calculate F1 score
pred_test <- as.array(model(as.matrix(data_test_enc[, -1])))
f1_score <- fbeta_score(as.numeric(data_test$Diabetes) -1, pred_test > 0.5)
cat("F1 score (neural network): ", f1_score, "\n")

################################################################################
#                             Local XAI Methods
################################################################################

# Create Converter
conv <- convert(model, input_dim = ncol(data_test_enc) - 1,
                input_names = colnames(data_test_enc)[-1])

# Use testdata
data <- data_test_enc

# Prediction-Sensitive Methods -------------------------------------------------

# Select instance
id <- 125
instance <- data[id, -1]

# Select feature
num <- 500
feature <- "Age"
feature_idx <- which(colnames(data)[-1] == feature)
grid <- seq(min(data[[feature]]), max(data[[feature]]), length.out = num)

# Combine instance with grid
x <- instance[rep(1, num), ]
x[[feature]] <- grid

# Calculate Gradient and SmoothGrad
grad <- get_result(run_grad(conv, x, ignore_last_act = FALSE), "data.frame")
grad <- grad[grad$feature == feature, c("value", "pred")]
grad$feature <- grid
sgrad <- get_result(run_smoothgrad(conv, x, ignore_last_act = FALSE, noise_level = 0.004), "data.frame")
sgrad <- sgrad[sgrad$feature == feature, c("value", "pred")]
sgrad$feature <- grid

# Create plot
p <- ggplot() +
  geom_labelline(mapping = aes(x = feature, y = pred), data = grad,
                 hjust = 0.95,
                 label = "Prediction", color = scales::muted("black")) +
  geom_labelline(mapping = aes(x = feature, y = value*30), data = grad,
                 hjust = 0.8,
                 label = "Gradient", color = scales::muted("blue")) +
  geom_labelsmooth(mapping = aes(x = feature, y = value*30), data = sgrad,
                   hjust = 0.2,
                   label = "SmoothGrad", color = scales::muted("red")) +
  geom_vline(xintercept = instance[[feature]], linetype = "dashed") +
  geom_texthline(yintercept = 0.5, label = "Threshold", color = "black",
                 hjust = 0.05, linewidth = 0.1) +
  theme_minimal(base_size = 14) +
  labs(x = "Age", y = "Prediction | Gradient * 30")

ggsave("figures/example_pred_sens.pdf", p, width = 7, height = 5)

# (Fixed-) Reference Methods ---------------------------------------------------
library(shapviz)

# Select instance
id <- 542
instance <- data[id, -1]

# Calculate IntGrad (zero baseline)
intgrad <- run_intgrad(conv, instance, ignore_last_act = FALSE, n = 200)

# Create plot using (an adopted version of) `shapviz`
pred <- get_result(intgrad, "data.frame")$pred[1]
pred_ref <- pred - get_result(intgrad, "data.frame")$decomp_goal[1]
S_deeplift <- matrix(get_result(intgrad)[,,1], nrow = 1)
colnames(S_deeplift) <- colnames(data)[-1]
X_deeplift <- signif(as.matrix(as.data.frame(instance)), digits = 3)
X_deeplift[1,1] <- if (X_deeplift[1,1] == 0) "female" else "male"
shap_deeplift <- shapviz(S_deeplift, X_deeplift, baseline = pred_ref)
p1 <- sv_force(shap_deeplift, annot_labels = c("f(x') = ", "f(x) = "),
               fill_colors = c("#FF0051", "#008BFB")) + ggtitle("Zero Baseline")

# Calculate IntGrad (reference baseline)
id_ref <- 2
intgrad_ref <- run_intgrad(conv, instance, x_ref = data[id_ref, -1],
                           ignore_last_act = FALSE, n = 200)

# Create plot using (an adopted version of) `shapviz`
pred <- get_result(intgrad_ref, "data.frame")$pred[1]
pred_ref <- pred - get_result(intgrad_ref, "data.frame")$decomp_goal[1]
S_deeplift <- matrix(get_result(intgrad_ref)[,,1], nrow = 1)
colnames(S_deeplift) <- colnames(data)[-1]
X_deeplift <- signif(as.matrix(as.data.frame(instance)), digits = 3)
X_deeplift[1,1] <- if (X_deeplift[1,1] == 0) "female" else "male"
X_deeplift_ref <- signif(as.matrix(as.data.frame(data[id_ref, -1])), digits = 3)
X_deeplift_ref[1,1] <- if (X_deeplift[1,1] == 0) "female" else "male"
X_deeplift[1,] <- paste0(X_deeplift[1,], " (vs. ", X_deeplift_ref[1,], ")")
shap_deeplift <- shapviz(S_deeplift, X_deeplift, baseline = pred_ref)
p2 <- sv_force(shap_deeplift, annot_labels = c("f(x') =", "f(x) = "),
               fill_colors = c("#FF0051", "#008BFB")) + ggtitle("Reference Patient")

ggsave("figures/example_ref.pdf", p1 / p2, width = 7, height = 5)


# Shapley-based ----------------------------------------------------------------
# Select instance
id <- 542
instance <- data[id, -1]

# Run GradSHAP
gradshap <- run_expgrad(conv, instance, data_ref = data[, -1],
                        ignore_last_act = FALSE, n = 10000)

# Create plot using (an adopted version of) `shapviz`
pred <- get_result(gradshap, "data.frame")$pred[1]
pred_ref <- pred - get_result(gradshap, "data.frame")$decomp_goal[1]
S_deeplift <- matrix(get_result(gradshap)[,,1], nrow = 1)
colnames(S_deeplift) <- colnames(data)[-1]
X_deeplift <- signif(as.matrix(as.data.frame(instance)), digits = 3)
X_deeplift[1,1] <- if (X_deeplift[1,1] == 0) "female" else "male"
shap_deeplift <- shapviz(S_deeplift, X_deeplift, baseline = pred_ref)
p1 <- sv_force(shap_deeplift, annot_labels = c("E[f(x)] =", "f(x) = "),
               fill_colors = c("#FF0051", "#008BFB"))

id <- 101
instance <- data[id, -1]

# Run GradSHAP
gradshap <- run_expgrad(conv, instance, data_ref = data[, -1],
                        ignore_last_act = FALSE, n = 10000)

# Create plot using (an adopted version of) `shapviz`
pred <- get_result(gradshap, "data.frame")$pred[1]
pred_ref <- pred - get_result(gradshap, "data.frame")$decomp_goal[1]
S_deeplift <- matrix(get_result(gradshap)[,,1], nrow = 1)
colnames(S_deeplift) <- colnames(data)[-1]
X_deeplift <- signif(as.matrix(as.data.frame(instance)), digits = 3)
X_deeplift[1,1] <- if (X_deeplift[1,1] == 0) "female" else "male"
shap_deeplift <- shapviz(S_deeplift, X_deeplift, baseline = pred_ref)
p2 <- sv_force(shap_deeplift, annot_labels = c("E[f(x)] =", "f(x) = "),
               fill_colors = c("#FF0051", "#008BFB"))

ggsave("figures/example_shapley.pdf", p1 / p2, width = 7, height = 5)

# Interaction-based ------------------------------------------------------------
# Select instance
id <- 277
id_ref <- 16
num <- 200

# Create torch tensors
x <- torch_tensor(as.matrix(data_enc[id, -1]), dtype = torch_float())
x_ref <- torch_tensor(as.matrix(data_enc[id_ref, -1]), dtype = torch_float())

# Interpolate between instance and reference
alpha <- expand.grid(seq(1/num, 1, length.out = num), seq(1/num, 1, length.out = num))
alpha <- torch_tensor(alpha[, 1] * alpha[, 2])$unsqueeze(-1)
input <- x_ref$repeat_interleave(as.integer(num^2), dim = 1) +
  alpha * (x - x_ref)$repeat_interleave(as.integer(num^2), dim = 1)
input$requires_grad <- TRUE

# Calculate IntegratedHessian values
out <- model(input)
grad <- autograd_grad(out$sum(), input, create_graph = TRUE)[[1]]
hessian <- torch_stack(lapply(seq_len(4), function(i) {
  autograd_grad(grad[, i]$sum(), input, create_graph = TRUE)[[1]]
}))
hessian <- torch_mean(hessian * alpha$unsqueeze(1), dim = 2) * (x - x_ref) *  (x - x_ref)$t()
intgrad <- get_result(run_intgrad(conv, x, x_ref = x_ref, ignore_last_act = FALSE, n = 200))[,,1]
res <- as.array(hessian)
res <- res  - diag(4) * diag(res)
main_effect <- intgrad - rowSums(res)
interaction_effect <- res + diag(main_effect)

# Plot result
dat <- expand.grid(x = colnames(data[, -1]), y = colnames(data[, -1]))
interaction_effect[upper.tri(interaction_effect)] <- NA
dat$fill <- c(interaction_effect)

p <- ggplot(dat) +
  geom_tile(aes(x = x, y = y, fill = fill), color = "white") +
  geom_label(aes(x = x, y = y, label = round(fill, 3)), color = "black") +
  scale_fill_gradient2(low = "#008BFB", mid = "white", high = "#FF0051",
                       na.value = "white",
                       limits = c(min(dat$fill, na.rm = TRUE), max(dat$fill, na.rm = TRUE)) ) +
  scale_y_discrete(limits = rev(colnames(data[, -1])), expand = c(0,0)) +
  scale_x_discrete(expand = c(0,0)) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  labs(fill = "Effect", x = NULL, y = NULL,
       title = paste0("f(x) = ", round(as.array(model(x)), 3),
                      " vs. f(x') = ", round(as.array(model(x_ref)), 3)))
ggsave("figures/example_interactions.pdf", p, width = 5, height = 4)


################################################################################
#                           From local to global
################################################################################
num <- 2500
instance <- data[sample(seq_len(nrow(data)), num), -1]

# Calculate global IntGrad and GradSHAP
intgrad <- run_intgrad(conv, instance, x_ref = data[23, -1], ignore_last_act = FALSE, n = 50)
gradshap <- run_expgrad(conv, instance, data_ref = data[, -1], ignore_last_act = FALSE, n = 100)
res <- data.table(rbind(
  data.frame(get_result(intgrad, "data.frame"), method = "IntGrad (global)"),
  data.frame(get_result(gradshap, "data.frame"), method = "GradSHAP (global)")
))

# Create plot
p1 <- ggplot(res) +
  geom_boxplot(aes(x = feature, y = abs(value), fill = method)) +
  theme_minimal() +
  ggsci::scale_fill_npg() +
  geom_hline(yintercept = 0) +
  theme(legend.position = "top") +
  ylim(0, 1.1) +
  labs(y = "Mean Absolute Contribution", x = NULL, fill = NULL)

# Calculate global PFI values using the `iml` package
pred_fun <- function(model, newdata) {
  as.array(model(torch_tensor(as.matrix(newdata))))
}
predictor <- Predictor$new(model, data = data[, -1],
                           y = data$Diabetes, predict.function = pred_fun)
imp <- FeatureImp$new(predictor, loss = "ce", compare = "difference")
imp$results$feature <- factor(imp$results$feature, levels = colnames(data)[-1])
p2 <- plot(imp, sort = FALSE) + coord_flip() + theme_minimal() + geom_vline(xintercept = 0)

# Combine and save plots
p <- p1 / p2 + plot_layout(heights = c(2/3, 1/3))
ggsave("figures/example_global.pdf", p, width = 6 , height = 6)


############################# Print session Info ###############################
sessionInfo()
