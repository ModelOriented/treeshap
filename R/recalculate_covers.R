#' Compute Cover values of your model for reference dataset other than the original set used to train model.
#'
#'
#' @param model Unified dataframe representation of the model created with a (model).unify function.
#' @param X Reference dataset. A dataframe with the same columns as in the training set of the model.
#'
#'
#' @return  Unified dataframe representation of the model as created with a (model).unify function,
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
#' unified <- gbm.unify(gbm_model)
#' recalculate_covers(unified, data[200:700, ])
#'}
recalculate_covers <- function(model, X) {
  # argument check
  if (!all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score") %in% colnames(model))) {
    stop("Given model dataframe is not a correct unified dataframe representation. Use (model).unify function.")
  }

  doesnt_work_with_NAs <- all(is.na(model$Missing)) #any(is.na(model$Missing) & !is.na(model$Feature)) #
  if (doesnt_work_with_NAs && any(is.na(x))) {
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

  return(model)
}
