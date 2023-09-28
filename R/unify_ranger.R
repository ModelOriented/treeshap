#' Unify ranger model
#'
#' Convert your ranger model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param rf_model An object of \code{ranger} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @import data.table
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
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#'
#'  library(ranger)
#'  data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'                             c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'                              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#'  data <- na.omit(cbind(data_fifa, target = fifa20$target))
#'
#'  rf <- ranger::ranger(target~., data = data, max.depth = 10, num.trees = 10)
#'  unified_model <- ranger.unify(rf, data)
#'  shaps <- treeshap(unified_model, data[1:2,])
#'  plot_contribution(shaps, obs = 1)
ranger.unify <- function(rf_model, data) {
  if(!'ranger' %in% class(rf_model)) {
    stop('Object rf_model was not of class "ranger"')
  }
  n <- rf_model$num.trees
  x <- lapply(1:n, function(tree) {
    tree_data <- data.table::as.data.table(ranger::treeInfo(rf_model, tree = tree))
    tree_data[, c("nodeID",  "leftChild", "rightChild", "splitvarName", "splitval", "prediction")]
  })
  return(ranger_unify.common(x = x, n = n, data = data, feature_names = rf_model$forest$independent.variable.names))
}


ranger_unify.common <- function(x, n, data, feature_names) {
  times_vec <- sapply(x, nrow)
  y <- data.table::rbindlist(x)
  y[, ("Tree") := rep(0:(n - 1), times = times_vec)]
  data.table::setnames(y, c("Node", "Yes", "No", "Feature", "Split",  "Prediction", "Tree"))
  y[, ("Feature") := as.character(get("Feature"))]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[, ("Missing") := NA]
  y$Cover <- 0
  y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
  y[is.na(get("Feature")), ("Decision.type") := NA]

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)

  # Here we lose "Quality" information
  y[!is.na(get("Feature")), ("Prediction") := NA]

  # treeSHAP assumes, that [prediction = sum of predictions of the trees]
  # in random forest [prediction = mean of predictions of the trees]
  # so here we correct it by adjusting leaf prediction values
  y[is.na(get("Feature")), ("Prediction") := I(get("Prediction") / n)]


  data.table::setcolorder(
    y, c("Tree", "Node", "Feature", "Decision.type", "Split",
         "Yes", "No", "Missing", "Prediction", "Cover"))

  data <- data[,colnames(data) %in% feature_names]

  ret <- list(model = as.data.frame(y), data = as.data.frame(data), feature_names = feature_names)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- FALSE
  attr(ret, "model") <- "ranger"
  return(set_reference_dataset(ret, as.data.frame(data)))
}
