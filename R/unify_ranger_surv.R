#' Unify ranger survival model - predicting mortality risk
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
#' \code{\link{catboost.unify}} for \code{\link[catboost:catboost.train]{Catboost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#'
#' library(ranger)
#' data_colon <- data.table::data.table(survival::colon)
#' data_colon <- na.omit(data_colon[get("etype") == 2, ])
#' surv_cols <- c("status", "time", "rx")
#'
#' feature_cols <- colnames(data_colon)[3:(ncol(data_colon) - 1)]
#'
#' train_x <- model.matrix(
#'   ~ -1 + .,
#'   data_colon[, .SD, .SDcols = setdiff(feature_cols, surv_cols[1:2])]
#' )
#' train_y <- survival::Surv(
#'   event = (data_colon[, get("status")] |>
#'              as.character() |>
#'              as.integer()),
#'   time = data_colon[, get("time")],
#'   type = "right"
#' )
#'
#' rf <- ranger::ranger(
#'   x = train_x,
#'   y = train_y,
#'   data = data_colon,
#'   max.depth = 10,
#'   num.trees = 10
#' )
#' unified_model <- ranger_surv.unify(rf, train_x)
#' shaps <- treeshap(unified_model, train_x[1:2,])
#'
ranger_surv.unify <- function(rf_model, data) {
  surv_common <- ranger_surv.common(rf_model, data)
  n <- surv_common$n
  chf_table_list <- surv_common$chf_table_list
  x <- lapply(chf_table_list, function(tree) {
    tree_data <- tree$tree_data
    nodes_chf <- tree$table
    tree_data$prediction <- rowSums(nodes_chf)
    tree_data[, c("nodeID", "leftChild", "rightChild", "splitvarName",
                  "splitval", "prediction")]
  })
  return(ranger_unify.common(x = x, n = n, data = data))
}

ranger_surv.common <- function(rf_model, data) {
  if (!"ranger" %in% class(rf_model)) {
    stop("Object rf_model was not of class \"ranger\"")
  }
  if (!"survival" %in% names(rf_model)) {
    stop("Object rf_model is not a survival random forest.")
  }
  n <- rf_model$num.trees
  chf_table_list <- lapply(1:n, function(tree) {
    tree_data <- data.table::as.data.table(ranger::treeInfo(rf_model,
                                                            tree = tree))

    # first get number of columns
    chf_node <- rf_model$forest$chf[[tree]]
    nodes_chf_n <- ncol(do.call(rbind, chf_node))
    nodes_prepare_chf_list <- lapply(
      X = chf_node,
      FUN = function(node) {
        if (identical(node, numeric(0L))) {
          rep(NA, nodes_chf_n)
        } else {
          node
        }
      }
    )
    list(table = do.call(rbind, nodes_prepare_chf_list), tree_data = tree_data)
  })
  return(list(chf_table_list = chf_table_list, n = n))
}

