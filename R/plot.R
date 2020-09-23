## plotting functions for treeshap package

shaps.plot_feature_importance <- function(shaps, max_vars = ncol(shaps), desc_sorting = TRUE,
                                          title = "Feature Importance", subtitle = "Mean absolute SHAP values of variables") {
  if (!is.logical(desc_sorting)) {
    stop("desc_sorting is not logical.")
  }
  if (!is.numeric(max_vars)) {
    stop("max_vars is not numeric.")
  }
  if (max_vars > ncol(shaps)) {
    warning("max_vars exceeded number of explained variables. All variables will be shown.")
    max_vars <- ncol(shaps)
  }

  mean <- colMeans(abs(shaps))
  df <- data.frame(variable = factor(names(mean)), importance = as.vector(mean))
  if (desc_sorting) {
    df$variable <- reorder(df$variable, df$importance)
  }
  df <- df[order(df$importance, decreasing = TRUE)[1:max_vars], ]

  # plot it
  pl <- ggplot(df, aes(x = variable, y = importance)) +
    geom_bar(stat = "identity", fill = DALEX::colors_discrete_drwhy(1))

  pl + coord_flip() +
    DALEX::theme_drwhy_vertical() +
    ylab("mean(|SHAP value|)") + xlab("") +
    labs(title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma) +
    theme(legend.position = "none")
}

#shaps.plot_feature_importance(shaps1, max_vars = 10)

#
