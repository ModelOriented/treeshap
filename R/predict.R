#' Predict
#'
#' Predict using unified_model representation.
#'
#' @param unified_model Unified model representation of the model created with a (model).unify function. \code{\link{model_unified.object}}
#' @param x Observations to predict. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model.
#'
#' @return a vector of predictions.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(gbm)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data['value_eur'] <- fifa20$target
#' gbm_model <- gbm::gbm(
#'              formula = value_eur ~ .,
#'              data = data,
#'              distribution = "laplace",
#'              n.trees = 1000,
#'              cv.folds = 2,
#'              interaction.depth = 2,
#'              n.cores = 1)
#' unified <- gbm.unify(gbm_model, data)
#' predict(unified, data[3:7, ])
#'}
predict.model_unified <- function(unified_model, x) {
  model <- unified_model$model

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

  if (!all(model$Feature %in% c(NA, colnames(x)))) {
    stop("Dataset does not contain all features ocurring in the model.")
  }

  # adapting model representation to C++ and extracting from dataframe to vectors
  roots <- which(model$Node == 0) - 1
  yes <- model$Yes - 1
  no <- model$No - 1
  missing <- model$Missing - 1
  is_leaf <- is.na(model$Feature)
  feature <- match(model$Feature, colnames(x)) - 1
  split <- model$Split
  value <- model$Prediction

  x <- as.data.frame(t(as.matrix(x)))
  is_na <- is.na(x) # needed, because dataframe passed to cpp somehow replaces missing values with random values

  predict_cpp(x, is_na, roots, yes, no, missing, is_leaf, feature, split, value)
}
