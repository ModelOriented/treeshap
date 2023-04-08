#' Unify ranger survival model
#'
#' Convert your ranger model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param rf_model An object of \code{ranger} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#' @param type A character to define the type of prediction. Either `"risk"` (default),
#'   which returns the cumulative hazards for each observation as risk score, or
#'   `"survival"`, which predicts the survival probability at certain time-points for each observation.
#' @param times A numeric vector of unique death times at which the prediction should be evaluated.
#'
#' @return For `type = "risk"` a unified model representation is returned - a \code{\link{model_unified.object}} object.
#'   For `type = "survival"` a list is returned that contains unified model representation ,
#'   (\code{\link{model_unified.object}} objects) for each time point. In this case, the list names are the
#'   `unique.death.times` (from the `ranger` object), at which the survival function was evaluated.
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
#' unified_model_risk <- ranger_surv.unify(rf, train_x, type = "risk")
#' shaps <- treeshap(unified_model_risk, train_x[1:2,])
#'
#'
#' unified_model_surv <- ranger_surv_fun.unify(rf, train_x, type = "survival")
#' # compute shaps for first 3 death times
#' for (m in unified_model_surv[1:3]) {
#'   shaps <- treeshap(m, train_x[1:2,])
#' }
#'
ranger_surv.unify <- function(rf_model, data, type = c("risk", "survival"), times = NULL) {
  type <- match.arg(type)
  stopifnot(ifelse(!is.null(times), is.numeric(times) && type == "survival", TRUE))
  surv_common <- ranger_surv.common(rf_model, data)
  n <- surv_common$n
  chf_table_list <- surv_common$chf_table_list

  if (type == "risk") {

    x <- lapply(chf_table_list, function(tree) {
      tree_data <- tree$tree_data
      nodes_chf <- tree$table
      tree_data$prediction <- rowSums(nodes_chf)
      tree_data[, c("nodeID", "leftChild", "rightChild", "splitvarName",
                    "splitval", "prediction")]
    })
    unified_return <- ranger_unify.common(x = x, n = n, data = data)

  } else if (type == "survival") {

    unique_death_times <- rf_model$unique.death.times

    if (is.null(times)) {
      compute_at_times <- unique_death_times
    } else {
      stepfunction <- stepfun(unique_death_times, c(unique_death_times[1], unique_death_times))
      compute_at_times <- stepfunction(times)
    }

    unified_return <- list()
    # iterate over time-points
    for (t in seq_len(length(compute_at_times))) {
      death_time <- compute_at_times[t]
      time_index <- which(unique_death_times == death_time)
      x <- lapply(chf_table_list, function(tree) {
        tree_data <- tree$tree_data
        nodes_chf <- tree$table[, time_index]

        # transform cumulative hazards to survival function
        # H(t) = -ln(S(t))
        # S(t) = exp(-H(t))
        tree_data$prediction <- exp(-nodes_chf)
        tree_data[, c("nodeID", "leftChild", "rightChild", "splitvarName",
                      "splitval", "prediction")]
      })
      unif <- ranger_unify.common(x = x, n = n, data = data)
      unified_return[[as.character(death_time)]] <- unif
    }
  }
  return(unified_return)
}

ranger_surv.common <- function(rf_model, data) {
  if (!"ranger" %in% class(rf_model)) {
    stop("Object rf_model was not of class \"ranger\"")
  }
  if (!"survival" %in% names(rf_model)) {
    stop("Object rf_model is not a random survival forest.")
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

