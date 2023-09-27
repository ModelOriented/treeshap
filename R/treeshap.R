#' Calculate SHAP values of a tree ensemble model.
#'
#' Calculate SHAP values and optionally SHAP Interaction values.
#'
#'
#' @param unified_model Unified data.frame representation of the model created with a (model).unify function. A \code{\link{model_unified.object}} object.
#' @param x Observations to be explained. A \code{data.frame} or \code{matrix} object with the same columns as in the training set of the model. Keep in mind that objects different than \code{data.frame} or plain \code{matrix} will cause an error or unpredictable behaviour.
#' @param interactions Whether to calculate SHAP interaction values. By default is \code{FALSE}. Basic SHAP values are always calculated.
#' @param verbose Whether to print progress bar to the console. Should be logical. Progress bar will not be displayed on Windows.
#'
#' @return A \code{\link{treeshap.object}} object. SHAP values can be accessed with \code{$shaps}. Interaction values can be accessed with \code{$interactions}.
#'
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
#' \code{\link{ranger_surv.unify}} for \code{ranger survival models}
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
#' treeshap1 <- treeshap(unified_model, head(data, 3))
#' plot_contribution(treeshap1, obs = 1)
#' treeshap1$shaps
#'
#' # It's possible to calcualte explanation over different part of the data set
#'
#' unified_model_rec <- set_reference_dataset(unified_model, data[1:1000, ])
#' treeshap_rec <- treeshap(unified_model, head(data, 3))
#' plot_contribution(treeshap_rec, obs = 1)
#'
#' # calculating SHAP interaction values
#' param2 <- list(objective = "reg:squarederror", max_depth = 7)
#' xgb_model2 <- xgboost::xgboost(as.matrix(data), params = param2, label = target, nrounds = 10)
#' unified_model2 <- xgboost.unify(xgb_model2, as.matrix(data))
#' treeshap2 <- treeshap(unified_model2, head(data, 3), interactions = TRUE)
#' treeshap2$interactions
#' }
treeshap <- function(unified_model, x, interactions = FALSE, verbose = TRUE) {
  model <- unified_model$model

  # argument check
  if (!("matrix" %in% class(x) | "data.frame" %in% class(x))) {
    stop("x parameter has to be data.frame or matrix.")
  }

  if (!is.model_unified(unified_model)) {
    stop("unified_model parameter has to of class model_unified. Produce it using *.unify function.")
  }

  if (!attr(unified_model, "missing_support") & any(is.na(x))) {
    stop("Given model does not work with missing values. Dataset x should not contain missing values.")
  }

  if (!all(model$Feature %in% c(NA, colnames(x)))) {
    stop("Dataset x does not contain all features occurring in the model.")
  }

  if (!all(colnames(x) %in% unique(model$Feature))) {
    stop("Dataset contains features not occurring in the model.")
  }

  if (attr(unified_model, "model") == "LightGBM" & !is.data.frame(x)) {
    stop("For LightGBM models data.frame object is required as x parameter. Please convert.")
  }

  if ((!is.numeric(verbose) & !is.logical(verbose)) | is.null(verbose)) {
    warning("Incorrect verbose argument, setting verbose = FALSE (progress will not be printed).")
    verbose <- FALSE
  }
  verbose <- verbose[1] > 0 # so verbose = numeric will work too
  x <- as.data.frame(x)

  # adapting model representation to C++ and extracting from dataframe to vectors
  roots <- which(model$Node == 0) - 1
  yes <- model$Yes - 1
  no <- model$No - 1
  missing <- model$Missing - 1
  feature <- match(model$Feature, colnames(x)) - 1
  split <- model$Split
  decision_type <- unclass(model$Decision.type)
  is_leaf <- is.na(model$Feature)
  value <- model$Prediction
  cover <- model$Cover

  x2 <- as.data.frame(sapply(x, as.numeric))
  if (nrow(x) > 1) x2 <- t(x2) # transposed to be able to pick a observation with [] operator in Rcpp
  is_na <- is.na(x2) # needed, because dataframe passed to cpp somehow replaces missing values with random values

  # calculating SHAP values
  if (interactions) {
    result <- treeshap_interactions_cpp(x2, is_na,
                                        roots, yes, no, missing, feature, split, decision_type, is_leaf, value, cover,
                                        verbose)
    shaps <- result$shaps
    interactions_array <- array(result$interactions,
                                dim = c(ncol(x), ncol(x), nrow(x)),
                                dimnames = list(colnames(x), colnames(x), rownames(x)))
  } else {
    shaps <- treeshap_cpp(x2, is_na,
                          roots, yes, no, missing, feature, split, decision_type, is_leaf, value, cover,
                          verbose)
    interactions_array <- NULL
  }

  dimnames(shaps) <- list(rownames(x), colnames(x))
  treeshap_obj <- list(shaps = as.data.frame(shaps), interactions = interactions_array,
                       unified_model = unified_model, observations = x)
  class(treeshap_obj) <- "treeshap"
  return(treeshap_obj)
}

#' treeshap results
#'
#' \code{treeshap} object produced by \code{treeshap} function.
#'
#' @return List consisting of four elements:
#' \describe{
#'   \item{shaps}{A \code{data.frame} with M columns, X rows (M - number of features, X - number of explained observations). Every row corresponds to SHAP values for a observation. }
#'   \item{interactions}{An \code{array} with dimensions (M, M, X) (M - number of features, X - number of explained observations). Every \code{[, , i]} slice is a symmetric matrix - SHAP Interaction values for a observation. \code{[a, b, i]} element is SHAP Interaction value of features \code{a} and \code{b} for observation \code{i}. Is \code{NULL} if interactions where not calculated (parameter \code{interactions} set \code{FALSE}.) }
#'   \item{unified_model}{An object of type \code{\link{model_unified.object}}. Unified representation of a model for which SHAP values were calculated. It is used by some of the plotting functions.}
#'   \item{observations}{Explained dataset. \code{data.frame} or \code{matrix}. It is used by some of the plotting functions.}
#' }
#'
#'
#' @seealso
#' \code{\link{treeshap}},
#'
#' \code{\link{plot_contribution}}, \code{\link{plot_feature_importance}}, \code{\link{plot_feature_dependence}}, \code{\link{plot_interaction}}
#'
#'
#' @name treeshap.object
NULL


#' Prints treeshap objects
#'
#' @param x a treeshap object
#' @param ... other arguments
#'
#' @export
#'
print.treeshap <- function(x, ...){
  print(x$shaps)
  if (!is.null(x$interactions)) {
    print(x$interactions)
  }
  return(invisible(NULL))
}

#' Check whether object is a valid treeshap object
#'
#' Does not check correctness of result, only basic checks
#'
#' @param x an object to check
#'
#' @return boolean
#'
#' @export
#'
is.treeshap <- function(x) {
  # class checks
  ("treeshap" %in% class(x)) &
    (is.data.frame(x$shaps)) &
    (is.null(x$interactions) | is.array(x$interactions)) &
    (is.model_unified(x$unified_model)) &
    (is.data.frame(x$observations) | is.matrix(x$observations)) &
    # dim checks
    (all(nrow(x$observations) == nrow(x$shaps)) & all(ncol(x$observations) == ncol(x$shaps))) &
    (is.null(x$interactions) | all(dim(x$interactions) == c(ncol(x$shaps), ncol(x$shaps), nrow(x$shaps)))) &
    # names check
    #all(rownames(x$observations) == rownames(x$shaps)) &
    all(colnames(x$observations) == colnames(x$shaps)) &
    (is.null(x$interactions) | all(dimnames(x$interactions)[[1]] == colnames(x$shaps)
                                & dimnames(x$interactions)[[2]] == colnames(x$shaps))) &
    #(is.null(x$interactions) | all(dimnames(x$interactions)[[3]] == rownames(x$shaps))) &
    # type check
    (is.null(x$interactions) | is.numeric(x$interactions)) &
    (is.numeric(as.matrix(x$shaps)))
}

