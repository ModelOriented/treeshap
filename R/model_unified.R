#' Unified model representation
#'
#' \code{model_unified} object produced by \code{*.unify} function.
#'
#' @return List consisting of two elements:
#'
#'
#' \strong{model} - A \code{data.frame} representing model with following columns:
#'
#' \item{Tree}{0-indexed ID of a tree}
#' \item{Node}{0-indexed ID of a node in a tree. In a tree the root always has ID 0}
#' \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#' \item{Decision.type}{A factor with two levels: "<" and "<=". In case of an internal node - predicate used for splitting observations. Otherwise - NA}
#' \item{Split}{For internal nodes threshold used for splitting observations. All observations that satisfy the predicate Decision.type(Split) ('< Split' / '<= Split') are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node. For leaves - NA}
#' \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes}
#' \item{No}{Index of a row containing a child Node}
#' \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#' \item{Prediction}{For leaves: Value of prediction in the leaf. For internal nodes: NA}
#' \item{Cover}{Number of observations seen by the internal node or collected by the leaf for the reference dataset}
#'
#' \strong{data} - Dataset used as a reference for calculating SHAP values. A dataset passed to the \code{*.unify} or \code{\link{set_reference_dataset}} function with \code{data} argument. A \code{data.frame}.
#'
#'
#' Object has two also attributes set:
#' \item{\code{model}}{A string. By what package the model was produced.}
#' \item{\code{missing_support}}{A boolean. Whether the model allows missing values to be present in explained dataset.}
#'
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{catboost.unify}} for \code{\link[catboost:catboost.train]{Catboost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#'
#' @name model_unified.object
#'
NULL

#' Prints model_unified objects
#'
#' @param x a model_unified object
#' @param ... other arguments
#'
#' @export
#'
#'
print.model_unified <- function(x, ...){
  print(x$model)
  return(invisible(NULL))
}

#' Check whether object is a valid model_unified object
#'
#' Does not check correctness of representation, only basic checks
#'
#' @param x an object to check
#'
#' @return boolean
#'
#' @export
#'
is.model_unified <- function(x) {
  # class checks
  ("model_unified" %in% class(x)) &
    is.data.frame(x$data) &
    is.data.frame(x$model) &
    # attributes check
    is.character(attr(x, "model")) &
    is.logical(attr(x, "missing_support")) &
    # colnames check
    all(c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover") %in% colnames(x$model)) &
    # column types check
    is.numeric(x$model$Tree) &
    is.numeric(x$model$Node) &
    is.character(x$model$Feature) &
    is.factor(x$model$Decision.type) &
    all(levels(x$model$Decision.type) == c("<=", "<")) &
    all(unclass(x$model$Decision.type) %in% c(1, 2, NA)) &
    is.numeric(x$model$Split) &
    is.numeric(x$model$Yes) &
    is.numeric(x$model$No) &
    (!attr(x, "missing_support") | is.numeric(x$model$Missing)) &
    is.numeric(x$model$Prediction) &
    is.numeric(x$model$Cover)
}
