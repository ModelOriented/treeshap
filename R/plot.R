## plotting functions for treeshap package

#' SHAP value based Feature Importance plot
#'
#' This function plots feature importance calculated as means of absolute values of SHAP values of variables (average impact on model output magnitude).
#'
#' @param shaps SHAP values dataframe produced with the \code{treeshap} function.
#' @param desc_sorting logical. Should the bars be sorted descending? By default TRUE.
#' @param max_vars maximum number of variables that shall be presented. By default all are presented.
#' @param title the plot's title, by default \code{'Feature Importance'}.
#' @param subtitle the plot's subtitle. By default no subtitle.
#'
#' @return a \code{ggplot2} object
#'
#' @export
#' @import ggplot2 DALEX
#' @importFrom stats reorder
#' @importFrom graphics text
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
#' unified_model <- xgboost.unify(xgb_model)
#' shaps <- treeshap(unified_model, head(data, 3))
#' plot_feature_importance(shaps, max_vars = 4)
#' }
plot_feature_importance <- function(shaps,
                                    desc_sorting = TRUE,
                                    max_vars = ncol(shaps),
                                    title = "Feature Importance",
                                    subtitle = NULL) {
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


#' SHAP value based Feature Dependence plot
#'
#' Depending on the value of a variable: how does it contribute into the prediction?
#'
#' @param shaps SHAP values dataframe produced with the \code{treeshap} function.
#' @param x dataset used to calculate \code{shaps}.
#' @param variable name or index of variable for which feature dependence will be plotted.
#' @param title the plot's title, by default \code{'Feature Dependence'}.
#' @param subtitle the plot's subtitle. By default no subtitle.
#'
#' @return a \code{ggplot2} object
#'
#' @export
#'
#' @import ggplot2
#' @importFrom DALEX theme_drwhy
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
#' unified_model <- xgboost.unify(xgb_model)
#' x <- head(data, 100)
#' shaps <- treeshap(unified_model, x)
#' plot_feature_dependence(shaps, x, variable = "overall")
#' }
plot_feature_dependence <- function(shaps, x, variable,
                                    title = "Feature Dependence", subtitle = NULL) {
  # TODO - add interactions as in https://christophm.github.io/interpretable-ml-book/shap.html

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


  df <- data.frame(var_value = x[[variable]], shap_value = shaps[[variable]])
  p <- ggplot(df, aes(x = var_value, y = shap_value)) +
    geom_point()

  p +
    theme_drwhy() +
    xlab(variable) + ylab(paste0("SHAP value for ", variable)) +
    labs(title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma)
}


#' SHAP value based Break-Down plot
#'
#' This function plots contributions of features into the prediction for a single observation.
#'
#' @param shap SHAP values dataframe produced with the \code{treeshap} function, containing only one row.
#' @param x \code{NULL} or dataframe with 1 observation used to calculate \code{shap}.
#' Used only for aesthetic reasons - to include observation values for a different variables next to the variable names in labels on the y axis.
#' By default is \code{NULL} and then labels on the y axis are just variable names.
#' @param model \code{NULL} or dataframe containing unified representation of explained model created with a (model).unify function.
#' Used to calculate mean prediction of the model to use as a baseline.
#' If \code{NULL} then baseline will be set as \code{0} and difference between individual prediction and model's mean prediction will be explained.
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
#' @importFrom DALEX theme_drwhy_vertical colors_breakdown_drwhy
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
#' unified_model <- xgboost.unify(xgb_model)
#' x <- head(data, 1)
#' shap <- treeshap(unified_model, x)
#' plot_contribution(shap, x, unified_model, min_max = c(0, 120000000))
#' }
plot_contribution <- function(shap,
                              x = NULL,
                              model = NULL,
                              max_vars = 5,
                              min_max = NA,
                              digits = 3,
                              title = "SHAP Break-Down",
                              subtitle = "") {
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
    mean_prediction <- sum(model[is_leaf, "Quality/Score"] * model$Cover[is_leaf]) / sum(model$Cover[is_root])
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
  df$prev[max_vars + 3] <- df$contribution[1]
  df$cumulative[max_vars + 3] <- df$cumulative[max_vars + 2]
  df$sign[max_vars + 3] <- "X"
  df$text[max_vars + 3] <- as.character(round(df$contribution[max_vars + 3], digits))

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


#' SHAP Interaction value plot
#'
#' This function plots SHAP Interaction value for two variables depending on the value of the first variable.
#' Value of the second variable is marked with the color.
#'
#' @param interactions SHAP Interactions array produced with \code{treeshap(interactions = TRUE)} function.
#' @param x dataset used to calculate \code{interactions}.
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
#' @importFrom DALEX theme_drwhy
#'
#' @seealso
#' \code{\link{treeshap}} for calculation of SHAP Interaction values
#'
#' @examples
#' \dontrun{
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param2 <- list(objective = "reg:squarederror", max_depth = 20)
#' xgb_model2 <- xgboost::xgboost(as.matrix(data), params = param2, label = target, nrounds = 10)
#' unified_model2 <- xgboost.unify(xgb_model2)
#' inters <- treeshap(unified_model2, data[1:50, ], interactions = TRUE)
#' plot_interaction(inters, data[1:50, ], "passing", "defending")
#' }
plot_interaction <- function(interactions, x, var1, var2,
                             title = "SHAP Interaction Value Plot",
                             subtitle = "") {
  # argument check
  if (dim(interactions)[3] != nrow(x)) {
    stop("Dataset x has different number of observations than interactions object.")
  }

  if (dim(interactions)[1] != ncol(x)) {
    stop("Dataset x has different number of variables than interactions object.")
  }

  if (is.character(var1)) {
    if (!(var1 %in% colnames(x))) stop("var1 is not a correct variable name. It does not occur in dataset x.")
    if (!(var1 %in% colnames(interactions))) stop("var1 is not a correct variable name. It does not occur in interactions object.")
  } else if (is.numeric(var1)) {
    if (var1 > ncol(x) || var1 < 1) stop("var1 is not a correct number.")
  }

  if (is.character(var2)) {
    if (!(var2 %in% colnames(x))) stop("var2 is not a correct variable name. It does not occur in dataset x.")
    if (!(var2 %in% colnames(interactions))) stop("var2 is not a correct variable name. It does not occur in interactions object.")
  } else if (is.numeric(var2)) {
    if (var2 > ncol(x) || var2 < 1) stop("var2 is not a correct number.")
  }


  interaction <- interactions[var1, var2, ]
  var1_value <- x[[var1]]
  var2_value <- x[[var2]]
  plot_data <- data.frame(var1_value = var1_value, var2_value = var2_value, interaction = interaction)

  x_lab <- ifelse(is.character(var1), var1, colnames(x)[var1])
  col_lab <- ifelse(is.character(var2), var2, colnames(x)[var])
  y_lab <- paste0("SHAP Interaction value for ", x_lab, " and ", col_lab)

  p <- ggplot(plot_data, aes(x = var1_value, y = interaction, color = var2_value)) +
    geom_point() +
    labs(x = x_lab, color = col_lab, y = y_lab, title = title, subtitle = subtitle) +
    scale_y_continuous(labels = scales::comma) +
    theme_drwhy()
  p
}
