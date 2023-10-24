#' Unify ranger survival model
#'
#' Convert your ranger model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @details
#' The survival forest implemented in the \code{ranger} package stores cumulative hazard
#' functions (CHFs) in the leaves of survival trees, as proposed for Random Survival Forests
#' (Ishwaran et al. 2008). The final model prediction is made by averaging these CHFs
#' from all the trees. To provide explanations in the form of a survival function,
#' the CHFs from the leaves are converted into survival functions (SFs) using
#' the formula SF(t) = exp(-CHF(t)).
#' However, it is important to note that averaging these SFs does not yield the correct
#' model prediction as the model prediction is the average of CHFs transformed in the same way.
#' Therefore, when you obtain explanations based on the survival function,
#' they are only proxies and may not be fully consistent with the model predictions
#' obtained using for example \code{predict} function.
#'
#
#' @param rf_model An object of \code{ranger} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#' @param type A character to define the type of model prediction to use. Either `"risk"` (default), which uses the risk score calculated as a sum of cumulative hazard function values, `"survival"`, which uses the survival probability at certain time-points for each observation, or `"chf"`, which used the cumulative hazard values at certain time-points for each observation.
#' @param times A numeric vector of unique death times at which the prediction should be evaluated. By default `unique.death.times` from model are used.
#'
#' @return For `type = "risk"` a unified model representation is returned - a \code{\link{model_unified.object}} object. For `type = "survival"` or `type = "chf"` - a \code{\link{model_unified_multioutput.object}} object is returned, which is a list that contains unified model representation (\code{\link{model_unified.object}} object) for each time point. In this case, the list names are time points at which the survival function was evaluated.
#'
#' @import data.table
#' @importFrom stats stepfun
#'
#' @export
#'
#' @seealso
#' \code{\link{ranger.unify}} for regression and classification \code{\link[ranger:ranger]{ranger models}}
#'
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
#' # compute shaps for 3 selected time points
#' unified_model_surv <- ranger_surv.unify(rf, train_x, type = "survival", times = c(23, 50, 73))
#' shaps_surv <- treeshap(unified_model_surv, train_x[1:2,])
#'
ranger_surv.unify <- function(rf_model, data, type = c("risk", "survival", "chf"), times = NULL) {
  type <- match.arg(type)

  stopifnot(
    "`times` must be a numeric vector and argument \
    `type = 'survival'` or `type = 'chf'` must be set." =
      ifelse(!is.null(times), is.numeric(times) && type == "survival", TRUE)
  )

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
    unified_return <- ranger_unify.common(x = x, n = n, data = data, feature_names = rf_model$forest$independent.variable.names)

  } else if (type == "survival" || type == "chf") {

    unique_death_times <- rf_model$unique.death.times

    if (is.null(times)) {
      compute_at_times <- unique_death_times
      # eval_times is required for list names (mainly when eval-times are
      # differing to the unique death times from the model as in the next case)
      eval_times <- as.character(compute_at_times)
    } else {
      stepfunction <- stepfun(unique_death_times, c(unique_death_times[1], unique_death_times))
      compute_at_times <- stepfunction(times)
      eval_times <- as.character(times)
    }

    # iterate over time-points
    unified_return <- lapply(compute_at_times, function(t) {
      time_index <- which(unique_death_times == t)
      x <- lapply(chf_table_list, function(tree) {
        tree_data <- tree$tree_data
        nodes_chf <- tree$table[, time_index]

        # transform cumulative hazards to survival function (if needed)
        # H(t) = -ln(S(t))
        # S(t) = exp(-H(t))
        tree_data$prediction <- if(type == "survival") exp(-nodes_chf) else nodes_chf
        tree_data[, c("nodeID", "leftChild", "rightChild", "splitvarName",
                      "splitval", "prediction")]
      })
      ranger_unify.common(x = x, n = n, data = data, feature_names = rf_model$forest$independent.variable.names)
    })
    names(unified_return) <- eval_times
    class(unified_return) <- "model_unified_multioutput"
  }
  return(unified_return)
}

ranger_surv.common <- function(rf_model, data) {
  if (!"ranger" %in% class(rf_model)) {
    stop("Object rf_model was not of class \"ranger\"")
  }
  if (!rf_model$treetype == "Survival") {
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

