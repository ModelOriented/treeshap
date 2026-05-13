#' Unify ranger model
#'
#' Convert your ranger model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param rf_model An object of \code{ranger} class. Categorical (factor) features are supported.
#' When the model is trained with \code{respect.unordered.factors = "ignore"} (the default for most
#' split rules), factor columns are treated as ordered by their integer codes and threshold-based
#' splits are used. When trained with \code{respect.unordered.factors = "partition"}, arbitrary
#' partitions of factor levels are used and represented as bitmask splits.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
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
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#' if (requireNamespace("ranger", quietly = TRUE)) {
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
#' }
ranger.unify <- function(rf_model, data) {
  if(!'ranger' %in% class(rf_model)) {
    stop('Object rf_model was not of class "ranger"')
  }
  n <- rf_model$num.trees
  x <- lapply(1:n, function(tree) {
    tree_data <- data.table::as.data.table(ranger::treeInfo(rf_model, tree = tree))
    # Fix for probability forests
    if (rf_model$treetype == "Probability estimation") {
      data.table::setnames(tree_data, "pred.1", "prediction")
    }
    tree_data[, c("nodeID",  "leftChild", "rightChild", "splitvarName", "splitval", "prediction")]
  })
  # Identify unordered (partition-mode) features from the forest object.
  # forest$is.ordered is a logical vector (one element per variable, same order as
  # independent.variable.names).  TRUE = ordered/numeric (threshold split),
  # FALSE = unordered factor (partition split, splitval is a comma-separated string
  # of right-child level indices from treeInfo).
  is_unordered <- if (!is.null(rf_model$forest$is.ordered)) {
    stats::setNames(!rf_model$forest$is.ordered, rf_model$forest$independent.variable.names)
  } else {
    NULL
  }
  return(ranger_unify.common(x = x, n = n, data = data,
                             feature_names = rf_model$forest$independent.variable.names,
                             is_unordered = is_unordered))
}


ranger_unify.common <- function(x, n, data, feature_names, is_unordered = NULL) {
  times_vec <- sapply(x, nrow)
  y <- data.table::rbindlist(x)
  y[, ("Tree") := rep(0:(n - 1), times = times_vec)]
  data.table::setnames(y, c("Node", "Yes", "No", "Feature", "Split",  "Prediction", "Tree"))
  y[, ("Feature") := as.character(get("Feature"))]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[, ("Missing") := NA]
  y$Cover <- 0

  # When any variable is unordered (partition mode), treeInfo returns split values
  # as character strings: comma-separated 1-based level indices going to the No
  # (right) child.  Ordered / numeric features in the same tree are then also
  # coerced to character by R's type system.
  has_char_split <- is.character(y$Split)
  has_partition <- has_char_split && !is.null(is_unordered) && any(is_unordered)

  if (has_partition) {
    # Identify internal nodes whose split feature is an unordered factor
    is_cat_split <- !is.na(y$Feature) & !is.na(y$Split) &
      (y$Feature %in% names(is_unordered)[is_unordered])

    if (any(is_cat_split)) {
      # Parse the comma-separated string of right-child level indices and compute
      # the right-group bitmask: bit (k-1) set means level k goes to the No child.
      bitmasks <- vapply(y$Split[is_cat_split], function(s) {
        right_levels <- as.integer(strsplit(s, ",", fixed = TRUE)[[1L]])
        sum(2^(right_levels - 1L))
      }, numeric(1L))
      y$Split[is_cat_split] <- as.character(bitmasks)
    }

    y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<", "=="))
    y[is.na(get("Feature")), ("Decision.type") := NA]
    if (any(is_cat_split)) {
      data.table::set(y, which(is_cat_split), "Decision.type", "==")
    }
  } else {
    y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
    y[is.na(get("Feature")), ("Decision.type") := NA]
  }

  # Convert character Split values to numeric where needed
  if (has_char_split) {
    y[, ("Split") := suppressWarnings(as.numeric(get("Split")))]
  }

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
