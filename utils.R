################################################################################
#                         Utility functions
################################################################################

# Neural Network (`torch` and `luz`) -------------------------------------------

# Define model
get_model <- nn_module(
  initialize = function(feat_in, feat_out = 1, dropout_rate = 0.3,
                        activation = "relu", classification = FALSE) {
    act <- switch (activation,
                   relu = nn_relu,
                   softplus = nn_softplus,
                   elu = nn_elu,
                   tanh = nn_tanh
    )

    if (classification) {
      self$net <- nn_sequential(
        nn_linear(feat_in, 256),
        act(),
        nn_dropout(p = dropout_rate),
        nn_linear(256, 128),
        act(),
        nn_dropout(p = dropout_rate),
        nn_linear(128, feat_out),
        nn_sigmoid()
      )
    } else {
      self$net <- nn_sequential(
        nn_linear(feat_in, 256),
        act(),
        nn_dropout(p = dropout_rate),
        nn_linear(256, 128),
        act(),
        nn_dropout(p = dropout_rate),
        nn_linear(128, feat_out)
      )
    }
  },

  forward = function(x) {
    self$net(x)
  }
)

# Create torch dataset
create_dataset <- dataset(
  initialize = function(df, y_name = "y") {
    feat_names <- setdiff(colnames(df), y_name)
    self$x <- as.matrix(df[, ..feat_names])
    self$y <- as.matrix(df[[y_name]])
  },

  .getitem = function(i) {
    list(x = torch_tensor(self$x[i, , drop = FALSE]),
         y = torch_tensor(self$y[i, , drop = FALSE]))

  },

  .length = function() {
    nrow(self$y)
  }
)

# `luz` metric for R²
luz_metric_r2 <- luz_metric(
  abbrev = "R2",
  initialize = function() {
    self$pred <- 0
    self$y <- 0
  },

  update = function(preds, target) {
    self$pred <- c(self$pred, as_array(preds$squeeze()))
    self$y <- c(self$y, as_array(target$squeeze()))
  },

  compute = function() {
    1 - mean((self$pred - self$y)**2) / mean((self$y - mean(self$y))**2)
  }
)


# Plotting ---------------------------------------------------------------------

# (small adoption of `shapviz::sv_force`)
sv_force <- function(object, row_id = 1L, max_display = 6L,
                             fill_colors = c("#f7d13d", "#a52c60"),
                             format_shap = getOption("shapviz.format_shap"),
                             format_feat = getOption("shapviz.format_feat"),
                             annot_labels = c("E[f(x)]=", "f(x)="),
                             contrast = TRUE, bar_label_size = 3.2,
                             show_annotation = TRUE, annotation_size = 3.2, ...) {
  stopifnot(
    "Exactly two fill colors must be passed" = length(fill_colors) == 2L,
    "format_shap must be a function" = is.function(format_shap),
    "format_feat must be a function" = is.function(format_feat)
  )
  object <- object[row_id, ]
  b <- get_baseline(object)
  dat <- .make_dat(object, format_feat = format_feat, sep = "=")
  if (ncol(object) > max_display) {
    dat <- .collapse(dat, max_display = max_display)
  }

  # Reorder rows and calculate order dependent columns
  .sorter <- function(y, p) {
    y <- y[order(abs(y$S)), ]
    y$to <- cumsum(y$S)
    y$from <- .lag(y$to, default = 0)
    hook <- y[nrow(y), "to"]
    vars <- c("to", "from")
    y[, vars] <- y[, vars] + p - hook
    y
  }
  dat$id <- "1"
  pred <- b + sum(dat$S)
  dat <- do.call(rbind, lapply(split(dat, dat$S >= 0), .sorter, p = pred))

  # Make a force plot
  b_pred <- c(b, pred)
  height <- grid::unit(0.17, "npc")

  p <- ggplot2::ggplot(
    dat,
    ggplot2::aes(
      xmin = from, xmax = to, y = id, fill = factor(S < 0, levels = c(FALSE, TRUE))
    )
  ) +
    gggenes::geom_gene_arrow(
      show.legend = FALSE,
      arrowhead_width = grid::unit(2, "mm"),
      arrow_body_height = height,
      arrowhead_height = height
    ) +
    ggrepel::geom_text_repel(
      ggplot2::aes(x = (from + to) / 2, y = as.numeric(id) + 0.08, label = label),
      size = bar_label_size,
      nudge_y = 0.3,
      segment.size = 0.1,
      segment.alpha = 0.5,
      direction = "both"
    ) +
    ggfittext::geom_fit_text(
      ggplot2::aes(label = paste0(ifelse(S > 0, "+", ""), format_shap(S))),
      show.legend = FALSE,
      contrast = contrast,
      ...
    ) +
    ggplot2::coord_cartesian(ylim = c(0.8, 1.2), clip = "off") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.13)) +
    # scale_y_discrete(expand = expansion(add = c(0.1 + 0.5 * show_annotation, 0.6))) +
    ggplot2::scale_fill_manual(values = fill_colors, drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      aspect.ratio = 1 / 4,
      panel.border = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank()
    ) +
    ggplot2::labs(y = ggplot2::element_blank(), x = "Prediction")

  if (show_annotation) {
    p <- p +
      ggplot2::annotate(
        "segment",
        x = b_pred,
        xend = b_pred,
        y = c(0.5, 0.75),
        yend = c(0.92, 1),
        linewidth = 0.3,
        linetype = 2
      ) +
      ggplot2::annotate(
        "text",
        x = b_pred,
        y = c(0.4, 0.65),
        label = paste0(annot_labels, format_shap(b_pred)),
        size = annotation_size
      )
  }
  p
}

# Helper functions for sv_waterfall() and sv_force()
.lag <- function(z, default = NA, lead = FALSE) {
  n <- length(z)
  if (n < 2L) {
    return(rep(default, times = n))
  }
  if (isTRUE(lead)) {
    return(c(z[2L:n], default))
  }
  c(default, z[1L:(n - 1L)])
}

## Turns "shapviz" object into a two-column data.frame
.make_dat <- function(object, format_feat, sep = " = ") {
  X <- get_feature_values(object)
  S <- get_shap_values(object)
  if (nrow(object) == 1L) {
    S <- drop(S)
    label <- paste(colnames(X), format_feat(X), sep = sep)
  } else {
    message("Aggregating SHAP values over ", nrow(object), " observations")
    S <- colMeans(S)
    J <- vapply(X, function(z) length(unique(z)) <= 1L, FUN.VALUE = TRUE)
    label <- colnames(X)
    if (any(J)) {
      label[J] <- paste(label[J], format_feat(X[1L, J]), sep = sep)
    }
  }
  data.frame(S = S, label = label)
}
