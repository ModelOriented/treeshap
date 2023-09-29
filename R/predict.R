#' Predict
#'
#' Predict using unified_model representation.
#'
#' @param object Unified model representation of the model created with a (model).unify function. \code{\link{model_unified.object}}
#' @param x Observations to predict. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model.
#' @param ... other parameters
#'
#' @return a vector of predictions.
#'
#' @export
#'
#' @examples
#' library(gbm)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data['value_eur'] <- fifa20$target
#' gbm_model <- gbm::gbm(
#'   formula = value_eur ~ .,
#'   data = data,
#'   distribution = "laplace",
#'   n.trees = 20,
#'   interaction.depth = 4,
#'   n.cores = 1)
#'   unified <- gbm.unify(gbm_model, data)
#'   predict(unified, data[2001:2005, ])
predict.model_unified <- function(object, x, ...) {
  unified_model <- object
  model <- unified_model$model
  x <- as.data.frame(x)

  # argument check
  if (!is.model_unified(unified_model)) {
    stop("unified_model parameter has to of class model_unified. Produce it using *.unify function.")
  }

  if (!("matrix" %in% class(x) | "data.frame" %in% class(x))) {
    stop("x parameter has to be data.frame or matrix.")
  }

  if (!attr(unified_model, "missing_support") && any(is.na(x))) {
    stop("Given model does not work with missing values. Dataset x should not contain missing values.")
  }

  x <- x[,colnames(x) %in% unified_model$feature_names]

  if (!all(model$Feature %in% c(NA, colnames(x)))) {
    stop("Dataset does not contain all features occurring in the model.")
  }

  # adapting model representation to C++ and extracting from dataframe to vectors
  roots <- which(model$Node == 0) - 1
  yes <- model$Yes - 1
  no <- model$No - 1
  missing <- model$Missing - 1
  is_leaf <- is.na(model$Feature)
  feature <- match(model$Feature, colnames(x)) - 1
  split <- model$Split
  decision_type <- unclass(model$Decision.type)
  #stopifnot(levels(decision_type) == c("<=", "<"))
  #stopifnot(all(decision_type %in% c(1, 2, NA)))
  value <- model$Prediction

  n <- nrow(x)
  x <- as.data.frame(sapply(x, as.numeric))
  if (n > 1) x <- t(x)

  is_na <- is.na(x) # needed, because dataframe passed to cpp somehow replaces missing values with random values

  predict_cpp(x, is_na, roots, yes, no, missing, is_leaf, feature, split, decision_type, value)
}
