# should be preceded with lgb.model.dt.tree
#' Unify LightGBM model
#'
#' Convert your LightGBM model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in the \code{treeshap()} function.
#'
#' @param lgb_model A lightgbm model - object of class \code{lgb.Booster}
#' @param data matrix for which calculations should be performed.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it FALSE if training data are used.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Prediction}{For leaves: Value of prediction in the leaf. For internal nodes: NA.}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#' @export
#' @import data.table
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' library(lightgbm)
#' param_lgbm <- list(objective = "regression", max_depth = 2,  force_row_wise = TRUE)
#' data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'              c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' data <- na.omit(cbind(data_fifa, fifa20$target))
#' sparse_data <- as.matrix(data[,-ncol(data)])
#' x <- lightgbm::lgb.Dataset(sparse_data, label = as.matrix(data[,ncol(data)]))
#' lgb_data <- lightgbm::lgb.Dataset.construct(x)
#' lgb_model <- lightgbm::lightgbm(data = lgb_data, params = param_lgbm, save_name = "", verbose = 0)
#' unified_model <- lightgbm.unify(lgb_model, sparse_data)
#' shaps <- treeshap(unified_model, data[1:2, ])
#' plot_contribution(shaps, obs = 1)
lightgbm.unify <- function(lgb_model, data, recalculate = FALSE) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("Package \"lightgbm\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  df <- lightgbm::lgb.model.dt.tree(lgb_model)
  stopifnot(c("split_index", "split_feature", "node_parent", "leaf_index", "leaf_parent", "internal_value",
              "internal_count", "leaf_value", "leaf_count")
            %in% colnames(df) )
  df <- data.table::as.data.table(df)
  #convert node_parent and leaf_parent into one parent column
  df[is.na(df$node_parent), "node_parent"] <- df[is.na(df$node_parent), "leaf_parent"]
  #convert values into one column...
  df[is.na(df$internal_value), "internal_value"] <- df[!is.na(df$leaf_value), "leaf_value"]
  #...and counts
  df[is.na(df$internal_count), "internal_count"] <- df[!is.na(df$leaf_count), "leaf_count"]
  df[["internal_count"]] <- as.numeric(df[["internal_count"]])
  #convert split_index and leaf_index into one column:
  max_split_index <- df[, max(split_index, na.rm = TRUE), tree_index]
  rep_max_split <- rep(max_split_index$V1, times = as.numeric(table(df$tree_index)))
  new_leaf_index <- rep_max_split + df[, "leaf_index"] + 1
  df[is.na(df$split_index), "split_index"] <- new_leaf_index[!is.na(new_leaf_index[["leaf_index"]]), 'leaf_index']
  df[is.na(df$split_gain), "split_gain"] <- df[is.na(df$split_gain), "leaf_value"]
  # On the basis of column 'Parent', create columns with childs: 'Yes', 'No' and 'Missing' like in the xgboost df:
  ret.first <- function(x) x[1]
  ret.second <- function(x) x[2]
  tmp <- data.table::merge.data.table(df[, .(node_parent, tree_index, split_index)], df[, .(tree_index, split_index, default_left, decision_type)],
                                      by.x = c("tree_index", "node_parent"), by.y = c("tree_index", "split_index"))
  y_n_m <- unique(tmp[, .(Yes = ifelse(decision_type %in% c("<=", "<"), ret.first(split_index),
                                       ifelse(decision_type %in% c(">=", ">"), ret.second(split_index), stop("Unknown decision_type"))),
                          No = ifelse(decision_type %in% c(">=", ">"), ret.first(split_index),
                                      ifelse(decision_type %in% c("<=", "<"), ret.second(split_index), stop("Unknown decision_type"))),
                          Missing = ifelse(default_left, ret.first(split_index),ret.second(split_index))),
                      .(tree_index, node_parent)])
  # POTENTIAL ISSUE WITH decision_type = "<" or ">"
  df <- data.table::merge.data.table(df[, c("tree_index", "depth", "split_index", "split_feature", "node_parent", "split_gain",
                                            "threshold", "internal_value", "internal_count")],
                                     y_n_m, by.x = c("tree_index", "split_index"),
                                     by.y = c("tree_index", "node_parent"), all.x = TRUE)
  df <- df[, c("tree_index", "split_index", "split_feature", "threshold", "Yes", "No", "Missing", "split_gain", "internal_count")]
  colnames(df) <- c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Prediction", "Cover")
  attr(df, "sorted") <- NULL

  ID <- paste0(df$Node, "-", df$Tree)
  df$Yes <- match(paste0(df$Yes, "-", df$Tree), ID)
  df$No <- match(paste0(df$No, "-", df$Tree), ID)
  df$Missing <- match(paste0(df$Missing, "-", df$Tree), ID)

  # Here we lose "Quality" information
  df$Prediction[!is.na(df$Feature)] <- NA

  # LightGBM calculates prediction as [mean_prediction + sum of predictions of trees]
  # treeSHAP assumes prediction are calculated as [sum of predictions of trees]
  # so here we adjust it
  #df[is.na(Feature), Prediction := Prediction + TODO]


  ret <- list(model = df, data = data)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "LightGBM"

  if (recalculate) {
    ret <- set_reference_dataset(ret, data)
  }

  return(ret)
}
