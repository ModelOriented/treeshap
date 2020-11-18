#' Unify xgboost model
#'
#' Convert your xgboost model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in the \code{treeshap()} function.
#'
#' @param xgb_model A xgboost model - object of class \code{xgb.Booster}
#' @param data matrix for which calculations should be performed.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it FALSE if training data are used.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA.}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Prediction}{For leaves: Value of prediction in the leaf. For internal nodes: NA.}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target,
#'                               nrounds = 200, verbose = 0)
#' unified_model <- xgboost.unify(xgb_model, as.matrix(data))
#' shaps <- treeshap(unified_model, data[1:2,])
#' plot_contribution(shaps, obs = 1)
#'
xgboost.unify <- function(xgb_model, data, recalculate = FALSE) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package \"xgboost\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  xgbtree <- xgboost::xgb.model.dt.tree(model = xgb_model)
  stopifnot(c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover") %in% colnames(xgbtree))
  xgbtree$Yes <- match(xgbtree$Yes, xgbtree$ID)
  xgbtree$No <- match(xgbtree$No, xgbtree$ID)
  xgbtree$Missing <- match(xgbtree$Missing, xgbtree$ID)
  xgbtree[xgbtree$Feature == 'Leaf', 'Feature'] <- NA
  xgbtree$Decision.type <- factor(x = rep("<=", times = nrow(xgbtree)), levels = c("<=", "<"))
  xgbtree$Decision.type[is.na(xgbtree$Feature)] <- NA
  xgbtree <- xgbtree[, c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Quality", "Cover")]
  colnames(xgbtree) <- c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover")

  # Here we lose "Quality" information
  xgbtree$Prediction[!is.na(xgbtree$Feature)] <- NA

  ret <- list(model = xgbtree, data = data)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "xgboost"

  if (recalculate) {
    ret <- set_reference_dataset(ret, data)
  }

  return(ret)
}


