#' SHAP value based Break-Down plot
#'
#' This function plots contributions of features into the prediction for a single observation.
#'
#' @param treeshap A treeshap object produced with the \code{treeshap} function.
#' @param obs A numeric indicating which observation should be plotted. Be deafult it's first observation.
#' @param max_vars maximum number of variables that shall be presented. Variables with the highest importance will be presented.
#' Remaining variables will be summed into one additional contribution. By default \code{5}.
#' @param min_max a range of OX axis. By default \code{NA}, therefore it will be extracted from the contributions of \code{x}.
#' But it can be set to some constants, useful if these plots are to be used for comparisons.
#' @param digits number of decimal places (\code{\link{round}}) to be used.
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
#' @examples
#' \dontrun{
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200)
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
                              title = "SHAP Break-Down",
                              subtitle = "") {

  shap <- treeshap$treeshap[obs,]
  model <- treeshap$unified_model
  x <- treeshap$observations[obs,]

  if (max_vars > ncol(shap)) {
    warning("max_vars exceeds number of variables. All variables will be shown.")
    max_vars <- ncol(shap)
  }
  if (nrow(shap) != 1) {
    warning("Only 1 observation can be plotted. Plotting 1st one.")
    shap <- shap[1, ]
  }

  # calculating model's mean prediction
  if (!is.null(model)) {
    if (!all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover") %in% colnames(model))) {
      stop("Given model dataframe is not a correct unified dataframe representation. Use (model).unify function.")
    }
    is_leaf <- is.na(model$Feature)
    is_root <- model$Node == 0
    mean_prediction <- sum(model[is_leaf, "Quality/Score"] * model$Cover[is_leaf]) / sum(model$Cover[is_root]) * sum(is_root)
  } else {
    mean_prediction <- 0
  }


  df <- data.frame(variable = colnames(shap), contribution = as.numeric(shap))

  # if using observation values, then setting variable names to showing their value
  if (!is.null(x)) {
    if (!all(colnames(x) == colnames(shap))) {
      stop("shap and x should have the same variables.")
    }
    if (nrow(x) != 1) {
      warning("Only 1 observation can be plotted. Assuming first observations in x and in shap are the same.")
      x <- x[1, ]
    }
    df$variable <- paste0(df$variable, " = ", as.character(x))
  }

  # selecting max_vars most important variables
  is_important <- order(abs(df$contribution), decreasing = TRUE)[1:max_vars]
  other_variables_contribution_sum <- sum(df$contribution[-is_important])
  df <- df[is_important, ]
  df$position <- 2:(max_vars + 1)
  if (max_vars < ncol(shap)) {
    df <- rbind(df, data.frame(variable = "+ all other variables",
                               contribution = other_variables_contribution_sum,
                               position = max_vars + 2))
  }

  # adding "prediction" bar
  df <- rbind(df, data.frame(variable = ifelse(is.null(model), "prediction deviation", "prediction"),
                             contribution = mean_prediction + sum(df$contribution),
                             position = max_vars + 3))

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
  df$prev[max_vars + 3] <- df$contribution[1] # or 0?
  df$cumulative[max_vars + 3] <- df$cumulative[max_vars + 2]
  df$sign[max_vars + 3] <- "X"
  df$text[max_vars + 3] <- as.character(round(df$contribution[max_vars + 3], digits))

  #removing intercept bar if no model passed
  if (is.null(model)) {
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
    geom_errorbarh(data = df[-(max_vars + 3), ],
                   aes(xmax = position - 0.85,
                       xmin = position + 0.85,
                       y = cumulative), height = 0,
                   color = "#371ea3") +
    geom_rect(alpha = 0.9) +
    geom_hline(data = df[df$variable == "intercept", ], aes(yintercept = contribution), lty = 3, alpha = 0.5, color = "#371ea3")

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
