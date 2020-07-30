#' Calculate SHAP values of a tree ensemble model.
#'
#' Check the structure of your ensemble model and calculate feature importance using \code{treeshap()} function.
#'
#'
#' @param model Unified dataframe representation of the model created with a (model).unify function.
#' @param x Observations to be explained. A dataframe with the same columns as in the training set of the model.
#'
#' @return SHAP values for given observations. A dataframe with the same columns as in the training set of the model.
#' Value from a column and a row is the SHAP value of the feature of the observation.
#'
#' @export
#'
#' @importFrom Rcpp sourceCpp
#' @useDynLib treeshap
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' @examples
#' \dontrun{
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200)
#' unified_model <- xgboost.unify(xgb_model)
#' treeshap(unified_model, head(data, 3))
#'}
treeshap <- function(model, x) {
  # argument check
  stopifnot(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover") %in% colnames(model))

  # adapting model representation to C++ and extracting from dataframe to vectors
  roots <- which(model$Node == 0) - 1
  yes <- model$Yes - 1
  no <- model$No - 1
  missing <- model$Missing - 1
  feature <- match(model$Feature, colnames(x)) - 1
  is_leaf <- is.na(model$Feature)
  value <- model[["Quality/Score"]]
  cover <- model$Cover

  # creating matrix containing information whether each observation fulfills each node split condition
  feature_columns <- feature + 1
  feature_columns[is.na(feature_columns)] <- 1
  fulfills <- t(t(x[, feature_columns]) <= model$Split)
  fulfills[, is.na(feature_columns)] <- NA

  # computing shaps
  shaps <- matrix(numeric(0), ncol = ncol(x))
  for (obs in 1:nrow(x)) {
    shaps_row <- treeshap_cpp(ncol(x), fulfills[obs, ], roots,
                                   yes, no, missing, feature, is_leaf, value, cover)
    shaps <- rbind(shaps, shaps_row)
  }

  colnames(shaps) <- colnames(x)
  rownames(shaps) <- c()
  return(as.data.frame(shaps))
}

