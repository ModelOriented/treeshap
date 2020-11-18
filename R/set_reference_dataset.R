#' Set reference dataset
#'
#' Change a dataset used as reference for calculating SHAP values.
#' Reference dataset is initially set with \code{data} argument in unifying function.
#' Usually reference dataset is dataset used to train the model.
#' Important property of reference dataset is that SHAPs for each observation add up to its deviation from mean prediction of reference dataset.
#'
#'
#' @param unified_model Unified model representation of the model created with a (model).unify function.
#' @param x Reference dataset. A dataframe with the same columns as in the training set of the model.
#'
#'
#' @return  Unified representation of the model as created with a (model).unify function,
#' but with Cover column containing updated values.
#'
#' @export
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
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
#' set_reference_dataset(unified, data[200:700, ])
#'}
set_reference_dataset <- function(unified_model, x) {
  model <- unified_model$model

  # argument check
  if (!all(c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction") %in% colnames(model))) {
    stop("Given model dataframe is not a correct unified dataframe representation. Use (model).unify function.")
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

  x <- as.data.frame(t(as.matrix(x)))
  is_na <- is.na(x) # needed, because dataframe passed to cpp somehow replaces missing values with random values

  model$Cover <- new_covers(x, is_na, roots, yes, no, missing, is_leaf, feature, split)

  ret <- list(model = model, data = x)
  #attributes(ret) <- attributes(model_unified)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- attr(unified_model, "missing_support")
  attr(ret, 'model') <- attr(unified_model, "model")

  ret

}
