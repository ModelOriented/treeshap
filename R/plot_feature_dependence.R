#' SHAP value based Feature Dependence plot
#'
#' Depending on the value of a variable: how does it contribute into the prediction?
#'
#' @param treeshap A treeshap object produced with the \code{\link{treeshap}} function. \code{\link{treeshap.object}}.
#' @param variable name or index of variable for which feature dependence will be plotted.
#' @param title the plot's title, by default \code{'Feature Dependence'}.
#' @param subtitle the plot's subtitle. By default no subtitle.
#'
#' @return a \code{ggplot2} object
#'
#' @export
#'
#' @import ggplot2
#'
#' @seealso
#' \code{\link{treeshap}} for calculation of SHAP values
#'
#' \code{\link{plot_contribution}}, \code{\link{plot_feature_importance}}, \code{\link{plot_interaction}}
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
#' x <- head(data, 100)
#' shaps <- treeshap(unified_model, x)
#' plot_feature_dependence(shaps, variable = "overall")
#' }
plot_feature_dependence <- function(treeshap, variable,
                                    title = "Feature Dependence", subtitle = NULL) {

  shaps <- treeshap$shaps
  x <- treeshap$observations

  # argument check
  if (!is.treeshap(treeshap)) {
    stop("treeshap parameter has to be correct object of class treeshap. Produce it using treeshap function.")
  }

  if (is.character(variable)) {
    if (!(variable %in% colnames(shaps))) {
      stop("Incorrect variable or shaps dataframe, variable should be one of variables in the shaps dataframe.")
    }
    if (!(variable %in% colnames(shaps))) {
      stop("Incorrect variable or x dataframe, varaible should be one of variables in the shaps dataframe.")
    }
  } else if (is.numeric(variable) && (length(variable) == 1)) {
    if (!all(colnames(shaps) == colnames(x))) {
      stop("shaps and x should have the same column names.")
    }
    if (!(variable %in% 1:ncol(shaps))) {
      stop("variable is an incorrect number.")
    }
    variable <- colnames(shaps)[variable]
  } else {
    stop("variable is of incorrect type.")
  }

  df <- data.frame(var_value = x[, variable], shap_value = shaps[, variable ])
  p <- ggplot(df, aes(x = var_value, y = shap_value)) +
    geom_point()

  p +
    theme_drwhy() +
    xlab(variable) + ylab(paste0("SHAP value for ", variable)) +
    labs(title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma)
}
