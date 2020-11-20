## plotting functions for treeshap package

#' SHAP value based Feature Importance plot
#'
#' This function plots feature importance calculated as means of absolute values of SHAP values of variables (average impact on model output magnitude).
#'
#' @param treeshap A treeshap object produced with the \code{\link{treeshap}} function. \code{\link{treeshap.object}}.
#' @param desc_sorting logical. Should the bars be sorted descending? By default TRUE.
#' @param max_vars maximum number of variables that shall be presented. By default all are presented.
#' @param title the plot's title, by default \code{'Feature Importance'}.
#' @param subtitle the plot's subtitle. By default no subtitle.
#'
#' @return a \code{ggplot2} object
#'
#' @export
#' @import ggplot2
#' @importFrom stats reorder
#' @importFrom graphics text
#'
#' @seealso
#' \code{\link{treeshap}} for calculation of SHAP values
#'
#' \code{\link{plot_contribution}}}, \code{\link{plot_feature_dependence}}}, \code{\link{plot_interaction}}}
#'
#'
#' @examples
#' \dontrun{
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200)
#' unified_model <- xgboost.unify(xgb_model, as.matrix(data))
#' shaps <- treeshap(unified_model, as.matrix(head(data, 3)))
#' plot_feature_importance(shaps, max_vars = 4)
#' }
plot_feature_importance <- function(treeshap,
                                    desc_sorting = TRUE,
                                    max_vars = ncol(shaps),
                                    title = "Feature Importance",
                                    subtitle = NULL) {
  shaps <- treeshap$shaps

  # argument check
  if (!is.treeshap(treeshap)) {
    stop("treeshap parameter has to be correct object of class treeshap. Produce it using treeshap function.")
  }

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
  df$variable <- reorder(df$variable, df$importance * ifelse(desc_sorting, 1, -1))
  df <- df[order(df$importance, decreasing = TRUE)[1:max_vars], ]

  p <- ggplot(df, aes(x = variable, y = importance)) +
    geom_bar(stat = "identity", fill = colors_discrete_drwhy(1))

  p + coord_flip() +
    theme_drwhy_vertical() +
    ylab("mean(|SHAP value|)") + xlab("") +
    labs(title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma) +
    theme(legend.position = "none")
}
