#' Calculate SHAP values of a tree ensemble model.
#'
#' Calculate SHAP values and optionally SHAP Interaction values.
#'
#'
#' @param unified_model Unified data.frame representation of the model created with a (model).unify function.
#' @param x Observations to be explained. A data.frame object with the same columns as in the training set of the model. Keep in mind that objects different than data.frame or plain matrix will cause an error or unpredictable behaviour.
#' @param interactions Whether to calculate SHAP interaction values. By default is \code{FALSE}.
#' @param verbose Wheter to print progress bar to the console.
#'
#' @return A treeshap object
#'
#' @export
#'
#' @importFrom Rcpp sourceCpp
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @useDynLib treeshap
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#' \code{\link{gbm.unify}} for \code{GBM models}
#' \code{\link{catboost.unify}} for \code{catboost models}
#' \code{\link{randomForest.unify}} for \code{randomForest models}
#' \code{\link{ranger.unify}} for \code{ranger models}
#'
#' @examples
#' \dontrun{
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#'
#' # calculating simple SHAP values
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200,
#'                               verbose = 0)
#' unified_model <- xgboost.unify(xgb_model, as.matrix(data))
#' shaps <- treeshap(unified_model, head(data, 3))
#' plot_contribution(shaps, obs = 1)
#'
#' # It's possible to calcualte explanation over different part of the data set
#'
#' unified_model_rec <- set_reference_dataset(unified_model, data[1:1000, ])
#' shaps_rec <- treeshap(unified_model, head(data, 3))
#' plot_contribution(shaps_rec, obs = 1)
#'
#' # calculating SHAP interaction values
#' param2 <- list(objective = "reg:squarederror", max_depth = 20)
#' xgb_model2 <- xgboost::xgboost(as.matrix(data), params = param2, label = target, nrounds = 10)
#' unified_model2 <- xgboost.unify(xgb_model2, as.matrix(data))
#' treeshap(unified_model2, head(data, 3), interactions = TRUE)
#' }
treeshap <- function(unified_model, x, interactions = FALSE, verbose = TRUE) {
  model <- unified_model$model

  # argument check
  if (!all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Prediction", "Cover") %in% colnames(model))) {
    stop("Given model dataframe is not a correct unified dataframe representation. Use (model).unify function.")
  }

  if (!attr(unified_model, "missing_support") && any(is.na(x))) {
    stop("Given model does not work with missing values. Dataset x should not contain missing values.")
  }

  if (!all(model$Feature %in% c(NA, colnames(x)))) {
    stop("Dataset x does not contain all features ocurring in the model.")
  }

  if (attr(unified_model, "model") == "LightGBM" & !is.data.frame(x)) {
    stop("For LightGBM models data.frame object is required as x parameter. Please convert.")
  }

  # adapting model representation to C++ and extracting from dataframe to vectors
  roots <- which(model$Node == 0) - 1
  yes <- model$Yes - 1
  no <- model$No - 1
  missing <- model$Missing - 1
  feature <- match(model$Feature, colnames(x)) - 1
  is_leaf <- is.na(model$Feature)
  value <- model$Prediction
  cover <- model$Cover

  # creating matrix containing information whether each observation fulfills each node split condition
  feature_columns <- feature + 1
  feature_columns[is.na(feature_columns)] <- 1
  fulfills <- t(t(x[, feature_columns]) <= model$Split)
  fulfills[, is.na(feature_columns)] <- NA

  # calculating SHAP values, and optionally SHAP Interaction values
  shaps <- matrix(numeric(0),
                  ncol = ncol(x), nrow = nrow(x),
                  dimnames = list(rownames(x), colnames(x)))
  if (interactions) {
    interactions_array <- array(numeric(0),
                                dim = c(ncol(x), ncol(x), nrow(x)),
                                dimnames = list(colnames(x), colnames(x), rownames(x)))
  } else {
    interactions_array <- NULL
  }

  if (verbose) {
    pb <- txtProgressBar(min = 0, max = nrow(x), initial = 0)
  }

  for (obs in 1:nrow(x)) {
    if (interactions) {
      interactions_result <- treeshap_interactions_cpp(ncol(x), fulfills[obs, ], roots, yes,
                                                      no, missing, feature, is_leaf, value, cover)
      interactions_slice <- interactions_result$interactions
      shaps_row <- interactions_result$shaps
    } else {
      shaps_row <- treeshap_cpp(ncol(x), fulfills[obs, ], roots,
                                yes, no, missing, feature, is_leaf, value, cover)
    }

    if (verbose) {
      setTxtProgressBar(pb, obs)
    }

    shaps[obs, ] <- shaps_row
    if (interactions) {
      interactions_array[, , obs] <- interactions_slice
    }
  }

  treeshap_obj <- list(shaps = as.data.frame(shaps), interactions = interactions_array,
                       unified_model = model, observations = x, data = unified_model$data)
  class(treeshap_obj) <- "treeshap"
  return(treeshap_obj)
}


#' Prints treeshap objects
#'
#' @param x a model_unified object
#' @param ... other arguments
#'
#' @export
#'
#'

print.treeshap <- function(x, ...){
  print(x$treeshap)
  return(invisible(NULL))
}

