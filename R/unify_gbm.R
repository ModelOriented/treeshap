#' Unify GBM model
#'
#' Convert your GBM model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param gbm_model An object of \code{gbm} class. At the moment, models built on data with categorical features
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
#'\donttest{
#' library(gbm)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data['value_eur'] <- fifa20$target
#' gbm_model <- gbm::gbm(
#'              formula = value_eur ~ .,
#'              data = data,
#'              distribution = "gaussian",
#'              n.trees = 50,
#'              interaction.depth = 4,
#'              n.cores = 1)
#' unified_model <- gbm.unify(gbm_model, data)
#' shaps <- treeshap(unified_model, data[1:2,])
#' plot_contribution(shaps, obs = 1)
#' }
gbm.unify <- function(gbm_model, data) {
  if(class(gbm_model) != 'gbm') {
    stop('Object gbm_model was not of class "gbm"')
  }
  if(any(gbm_model$var.type > 0)) {
    stop('Models built on data with categorical features are not supported - please encode them before training.')
  }
  x <- lapply(gbm_model$trees, data.table::as.data.table)
  times_vec <- sapply(x, nrow)
  y <- data.table::rbindlist(x)
  data.table::setnames(y, c("Feature", "Split", "Yes",
                            "No", "Missing", "ErrorReduction", "Cover",
                            "Prediction"))
  y[["Tree"]] <- rep(0:(length(gbm_model$trees) - 1), times = times_vec)
  y[["Node"]] <- unlist(lapply(times_vec, function(x) 0:(x - 1)))
  y <- y[, Feature := as.character(Feature)]
  y[y$Feature < 0, "Feature"] <- NA
  y[!is.na(y$Feature), "Feature"] <- attr(gbm_model$Terms, "term.labels")[as.integer(y[["Feature"]][!is.na(y$Feature)]) + 1]
  y[is.na(y$Feature), "ErrorReduction"] <- y[is.na(y$Feature), "Split"]
  y[is.na(y$Feature), "Split"] <- NA
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[y$Missing < 0, "Missing"] <- NA
  y <- y[, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "ErrorReduction", "Cover")]
  colnames(y) <- c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover")
  attr(y, "model") <- "gbm"

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y$Missing <- match(paste0(y$Missing, "-", y$Tree), ID)

  # GBM calculates prediction as [initF + sum of predictions of trees]
  # treeSHAP assumes prediction are calculated as [sum of predictions of trees]
  # so here we adjust it
  y[is.na(Feature), `Quality/Score` := `Quality/Score` + gbm_model$initF]


  ret <- list(model = y, data = data)
  class(ret) <- "model_unified"


  # Original covers in gbm_model are not correct
  set_reference_dataset(ret, data)


}
