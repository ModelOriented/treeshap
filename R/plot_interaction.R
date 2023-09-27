#' SHAP Interaction value plot
#'
#' This function plots SHAP Interaction value for two variables depending on the value of the first variable.
#' Value of the second variable is marked with the color.
#'
#' @param treeshap A treeshap object produced with \code{\link{treeshap}(interactions = TRUE)} function. \code{\link{treeshap.object}}.
#' @param var1 name or index of the first variable - plotted on x axis.
#' @param var2 name or index of the second variable - marked with color.
#' @param title the plot's title, by default \code{'SHAP Interaction Value Plot'}.
#' @param subtitle the plot's subtitle. By default no subtitle.
#'
#' @return a \code{ggplot2} object
#'
#' @export
#'
#' @import ggplot2
#'
#' @seealso
#' \code{\link{treeshap}} for calculation of SHAP Interaction values
#'
#' \code{\link{plot_contribution}}, \code{\link{plot_feature_importance}}, \code{\link{plot_feature_dependence}}
#'
#'
#' @examples
#' \donttest{
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param2 <- list(objective = "reg:squarederror", max_depth = 5)
#' xgb_model2 <- xgboost::xgboost(as.matrix(data), params = param2, label = target, nrounds = 10)
#' unified_model2 <- xgboost.unify(xgb_model2, data)
#' inters <- treeshap(unified_model2, as.matrix(data[1:50, ]), interactions = TRUE)
#' plot_interaction(inters, "dribbling", "defending")
#' }
plot_interaction <- function(treeshap, var1, var2,
                             title = "SHAP Interaction Value Plot",
                             subtitle = "") {

  interactions <- treeshap$interactions
  x <- treeshap$observations

  # argument check
  if (!is.treeshap(treeshap)) {
    stop("treeshap parameter has to be correct object of class treeshap. Produce it using treeshap function.")
  }

  if (is.null(interactions)) {
    stop("SHAP Interaction values were not calculated in treeshap object. You need to use treeshap(interactions = TRUE).")
  }

  if (is.character(var1)) {
    if (!(var1 %in% colnames(x))) stop("var1 is not a correct variable name. It does not occur in the dataset.")
    if (!(var1 %in% colnames(interactions))) stop("var1 is not a correct variable name. It does not occur in interactions object.")
  } else if (is.numeric(var1)) {
    if (var1 > ncol(x) || var1 < 1) stop("var1 is not a correct number.")
  }

  if (is.character(var2)) {
    if (!(var2 %in% colnames(x))) stop("var2 is not a correct variable name. It does not occur in the dataset.")
    if (!(var2 %in% colnames(interactions))) stop("var2 is not a correct variable name. It does not occur in interactions object.")
  } else if (is.numeric(var2)) {
    if (var2 > ncol(x) || var2 < 1) stop("var2 is not a correct number.")
  }


  interaction <- interactions[var1, var2, ]
  var1_value <- x[,var1]
  var2_value <- x[,var2]
  plot_data <- data.frame(var1_value = var1_value, var2_value = var2_value, interaction = interaction)

  x_lab <- ifelse(is.character(var1), var1, colnames(x)[var1])
  col_lab <- ifelse(is.character(var2), var2, colnames(x)[var2])
  y_lab <- paste0("SHAP Interaction value for ", x_lab, " and ", col_lab)

  p <- ggplot(plot_data, aes(x = var1_value, y = interaction, color = var2_value)) +
    geom_point() +
    labs(x = x_lab, color = col_lab, y = y_lab, title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma) +
    theme_drwhy()
  p
}
