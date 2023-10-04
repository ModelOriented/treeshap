#' SHAP value based Break-Down plot
#'
#' This function plots contributions of features into the prediction for a single observation.
#'
#' @param treeshap A treeshap object produced with the \code{\link{treeshap}} function. \code{\link{treeshap.object}}.
#' @param obs A numeric indicating which observation should be plotted. Be default it's first observation.
#' @param max_vars maximum number of variables that shall be presented. Variables with the highest importance will be presented.
#' Remaining variables will be summed into one additional contribution. By default \code{5}.
#' @param min_max a range of OX axis. By default \code{NA}, therefore it will be extracted from the contributions of \code{x}.
#' But it can be set to some constants, useful if these plots are to be used for comparisons.
#' @param digits number of decimal places (\code{\link{round}}) to be used.
#' @param explain_deviation if \code{TRUE} then instead of explaining prediction and plotting intercept bar, only deviation from mean prediction of the reference dataset will be explained. By default \code{FALSE}.
#' @param title the plot's title, by default \code{'SHAP Break-Down'}.
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
#' \code{\link{plot_feature_importance}}, \code{\link{plot_feature_dependence}}, \code{\link{plot_interaction}}
#'
#'
#' @examples
#' \donttest{
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target,
#'                               nrounds = 20, verbose = FALSE)
#' unified_model <- xgboost.unify(xgb_model, as.matrix(data))
#' x <- head(data, 1)
#' shap <- treeshap(unified_model, x)
#' plot_contribution(shap, 1,  min_max = c(0, 120000000))
#' }
plot_contribution <- function(treeshap,
                              obs = 1,
                              max_vars = 5,
                              min_max = NA,
                              digits = 3,
                              explain_deviation = FALSE,
                              title = "SHAP Break-Down",
                              subtitle = "") {

  shap <- treeshap$shaps[obs, ]
  model <- treeshap$unified_model$model
  x <- treeshap$observations[obs, ]

  # argument check
  if (!is.treeshap(treeshap)) {
    stop("treeshap parameter has to be correct object of class treeshap. Produce it using treeshap function.")
  }

  if (max_vars > ncol(shap)) {
    warning("max_vars exceeds number of variables. All variables will be shown.")
    max_vars <- ncol(shap)
  }
  if (nrow(shap) != 1) {
    warning("Only 1 observation can be plotted. Plotting 1st one.")
    shap <- shap[1, ]
  }

  # setting intercept
  mean_prediction <- mean(predict.model_unified(treeshap$unified_model, treeshap$unified_model$data))
  if (explain_deviation) {
    mean_prediction <- 0
  }

  df <- data.frame(variable = colnames(shap), contribution = as.numeric(shap))

  # setting variable names to showing their value
  df$variable <- paste0(df$variable, " = ", as.character(x))

  # selecting max_vars most important variables
  is_important <- order(abs(df$contribution), decreasing = TRUE)[1:max_vars]
  other_variables_contribution_sum <- sum(df$contribution[-is_important])
  df <- df[is_important, ]
  df$position <- 2:(max_vars + 1)
  if (max_vars < ncol(shap)) {
    df <- rbind(df, data.frame(variable = "+ all other variables",
                               contribution = other_variables_contribution_sum,
                               position = max(df$position) + 1))
  }

  # adding "prediction" bar
  df <- rbind(df, data.frame(variable = ifelse(explain_deviation, "prediction deviation", "prediction"),
                             contribution = mean_prediction + sum(df$contribution),
                             position = max(df$position) + 1))

  df$sign <- ifelse(df$contribution >= 0, "1", "-1")

  # adding "intercept" bar
  df <- rbind(df, data.frame(variable = "intercept",
                             contribution = mean_prediction,
                             position = 1,
                             sign = "X"))

  # ordering
  df <- df[order(df$position), ]

  # adding columns needed by plot
  df$cumulative <- cumsum(df$contribution)
  df$prev <- df$cumulative - df$contribution
  df$text <- as.character(round(df$contribution, digits))
  df$text[df$contribution > 0] <- paste0("+", df$text[df$contribution > 0])

  # intercept bar corrections:
  df$prev[1] <- df$contribution[1]
  df$text[1] <- as.character(round(df$contribution[1], digits))

  # prediction bar corrections:
  df$prev[nrow(df)] <- df$contribution[1]
  df$cumulative[nrow(df)] <- df$cumulative[max_vars + 2]
  if (!explain_deviation) { #  assuring it doesn't differ from prediction because of some numeric errors
    df$cumulative[nrow(df)] <- predict.model_unified(treeshap$unified_model, x)
  }
  df$sign[nrow(df)] <- "X"
  df$text[nrow(df)] <- as.character(round(df$contribution[nrow(df)], digits))

  # removing intercept bar if requested by explain_deviation argument
  if (explain_deviation) {
    df <- df[-1, ]
  }

  # reversing postions to sort bars decreasing
  df$position <- rev(df$position)

  # base plot
  p <- ggplot(df, aes(x = position + 0.5,
                      y = pmax(cumulative, prev),
                      xmin = position + 0.15, xmax = position + 0.85,
                      ymin = cumulative, ymax = prev,
                      fill = sign,
                      label = text))

  # add rectangles and hline
  p <- p +
    geom_errorbarh(data = df[-c(nrow(df), if (explain_deviation) nrow(df) - 1), ],
                   aes(xmax = position - 0.85,
                       xmin = position + 0.85,
                       y = cumulative), height = 0,
                   color = "#371ea3") +
    geom_rect(alpha = 0.9) +
    if (!explain_deviation) (geom_hline(data = df[df$variable == "intercept", ],
                                        aes(yintercept = contribution),
                                        lty = 3, alpha = 0.5, color = "#371ea3"))


  # add adnotations
  drange <- diff(range(df$cumulative))
  p <- p + geom_text(aes(y = pmax(cumulative,  cumulative - contribution)),
                     vjust = 0.5,
                     nudge_y = drange * 0.05,
                     hjust = 0,
                     color = "#371ea3")

  # set limits for contributions
  if (any(is.na(min_max))) {
    x_limits <- scale_y_continuous(expand = c(0.05, 0.15), name = "", labels = scales::comma)
  } else {
    x_limits <- scale_y_continuous(expand = c(0.05, 0.15), name = "", limits = min_max, labels = scales::comma)
  }

  p <- p + x_limits +
    scale_x_continuous(labels = df$variable, breaks = df$position + 0.5, name = "") +
    scale_fill_manual(values = colors_breakdown_drwhy())

  # add theme
  p + coord_flip() + theme_drwhy_vertical() +
    theme(legend.position = "none") +
    labs(title = title, subtitle = subtitle)
}
