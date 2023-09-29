#' Unify XGBoost model
#'
#' Convert your XGBoost model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param xgb_model A XGBoost model - object of class \code{xgb.Booster}
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it \code{FALSE} if training data are used.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target,
#'                               nrounds = 20, verbose = 0)
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
  xgbtree[is.na(xgbtree$Split), 'Feature'] <- NA
  xgbtree$Decision.type <- factor(x = rep("<=", times = nrow(xgbtree)), levels = c("<=", "<"))
  xgbtree$Decision.type[is.na(xgbtree$Feature)] <- NA
  xgbtree <- xgbtree[, c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Quality", "Cover")]
  colnames(xgbtree) <- c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover")

  # Here we lose "Quality" information
  xgbtree$Prediction[!is.na(xgbtree$Feature)] <- NA

  feature_names <- xgb_model$feature_names
  data <- data[,colnames(data) %in% feature_names]

  ret <- list(model = as.data.frame(xgbtree), data = as.data.frame(data), feature_names = feature_names)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "xgboost"

  if (recalculate) {
    ret <- set_reference_dataset(ret, as.data.frame(data))
  }

  return(ret)
}
