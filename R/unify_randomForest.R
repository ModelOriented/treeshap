#' Unify randomForest model
#'
#' Convert your randomForest model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param rf_model An object of \code{randomForest} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data data.frame for which calculations should be performed.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Othwerwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: the reduction in the loss function as a result of splitting this node.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#'
#' @import data.table
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
#'
#' library(randomForest)
#' data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'                            c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'                              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' data <- na.omit(cbind(data_fifa, target = fifa20$target))
#'
#' rf <- randomForest::randomForest(target~., data = data, maxnodes = 10, ntree = 10)
#' unified_model <- randomForest.unify(rf, data)
#' shaps <- treeshap(unified_model, data[1:2,])
#' # plot_contribution(shaps, obs = 1)
#'
randomForest.unify <- function(rf_model, data) {
  if(!'randomForest' %in% class(rf_model)){stop('Object rf_model was not of class "randomForest"')}
  if(any(attr(rf_model$terms, "dataClasses") != "numeric")) {
    stop('Models built on data with categorical features are not supported - please encode them before training.')
  }
  n <- rf_model$ntree
  ret <- data.table()
  x <- lapply(1:n, function(tree){
    tree_data <- as.data.table(randomForest::getTree(rf_model, k = tree, labelVar = TRUE))
    tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
  })
  times_vec <- sapply(x, nrow)
  y <- rbindlist(x)
  y[, Tree := rep(0:(n - 1), times = times_vec)]
  y[, Node := unlist(lapply(times_vec, function(x) 0:(x - 1)))]
  setnames(y, c("Yes", "No", "Feature", "Split",  "Quality/Score", "Tree", "Node"))
  y[, Feature := as.character(Feature)]
  y[, Yes := Yes - 1]
  y[, No := No - 1]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[, Missing := NA]
  attr(y, "model") <- "randomForest"

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y[, Missing := Yes]
  y$Cover <- 0

  # treeSHAP assumes, that [prediction = sum of predictions of the trees]
  # in random forest [prediction = mean of predictions of the trees]
  # so here we correct it by adjusting leaf prediction values
  y[is.na(Feature), `Quality/Score` := `Quality/Score` / n]


  setcolorder(y, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover"))

  ret <- list(model = y, data = data)
  class(ret) <- "model_unified"
  set_reference_dataset(ret, data)
}
