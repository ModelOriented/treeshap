#' Set reference dataset
#'
#' Change a dataset used as reference for calculating SHAP values.
#' Reference dataset is initially set with \code{data} argument in unifying function.
#' Usually reference dataset is dataset used to train the model.
#' Important property of reference dataset is that SHAP values for each observation add up to its deviation from mean prediction for a reference dataset.
#'
#'
#' @param unified_model Unified model representation of the model created with a (model).unify function. (\code{\link{model_unified.object}}).
#' @param x Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model.
#'
#' @return  \code{\link{model_unified.object}}. Unified representation of the model as created with a (model).unify function,
#' but with changed reference dataset (Cover column containing updated values).
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
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
  data <- x

  # argument check
  if (!("matrix" %in% class(x) | "data.frame" %in% class(x))) {
    stop("x parameter has to be data.frame or matrix.")
  }

  if (!("model_unified" %in% class(unified_model))) {
    stop("unified_model parameter has to of class model_unified. Produce it using *.unify function.")
  }

  if (!all(c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction") %in% colnames(model))) {
    stop("Given model dataframe is not a correct unified dataframe representation. Use (model).unify function.")
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

  n <- nrow(x)
  x <- as.data.frame(sapply(x, as.numeric))
  if (n > 1) x <- t(x)
  is_na <- is.na(x) # needed, because dataframe passed to cpp somehow replaces missing values with random values

  model$Cover <- new_covers(x, is_na, roots, yes, no, missing, is_leaf, feature, split, decision_type)

  ret <- list(model = as.data.frame(model), data = as.data.frame(data), feature_names = unified_model$feature_names)
  #attributes(ret) <- attributes(model_unified)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- attr(unified_model, "missing_support")
  attr(ret, 'model') <- attr(unified_model, "model")

  return(ret)
}
