#' Unify GBM model
#'
#' Convert your GBM model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param gbm_model An object of \code{gbm} class. Categorical (factor) features are supported.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#' \donttest{
#' if (requireNamespace("gbm", quietly = TRUE)) {
#'   library(gbm)
#'   data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#'   data['value_eur'] <- fifa20$target
#'   gbm_model <- gbm::gbm(
#'                formula = value_eur ~ .,
#'                data = data,
#'                distribution = "gaussian",
#'                n.trees = 20,
#'                interaction.depth = 4,
#'                n.cores = 1)
#'   unified_model <- gbm.unify(gbm_model, data)
#'   shaps <- treeshap(unified_model, data[1:2,])
#'   plot_contribution(shaps, obs = 1)
#' }}
gbm.unify <- function(gbm_model, data) {
  if(!inherits(gbm_model,'gbm')) {
    stop('Object gbm_model was not of class "gbm"')
  }
  has_cat <- any(gbm_model$var.type > 0)
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

  # For categorical features, replace the c.splits index stored in Split with a
  # bitmask encoding which factor levels go to the No (right) child.
  # Bit k-1 set in the bitmask means factor level k (1-based) goes right.
  if (has_cat) {
    cat_var_0based <- which(gbm_model$var.type > 0) - 1L
    is_cat_split <- !is.na(y$Feature) & (as.integer(y$Feature) %in% cat_var_0based)
    if (any(is_cat_split)) {
      cat_split_indices <- as.integer(y$Split[is_cat_split]) + 1L  # convert 0-based split index to 1-based R list index
      bitmasks <- vapply(cat_split_indices, function(split_idx) {
        c_split <- gbm_model$c.splits[[split_idx]]
        # c_split[k] == 1 means level k goes to the No (right) child
        right_levels <- which(c_split == 1L)
        if (length(right_levels) == 0L) return(0)
        sum(2^(right_levels - 1))
      }, numeric(1))
      y$Split[is_cat_split] <- bitmasks
    }
  }

  y[!is.na(y$Feature), "Feature"] <- attr(gbm_model$Terms, "term.labels")[as.integer(y[["Feature"]][!is.na(y$Feature)]) + 1]
  y[is.na(y$Feature), "ErrorReduction"] <- y[is.na(y$Feature), "Split"]
  y[is.na(y$Feature), "Split"] <- NA
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[y$Missing < 0, "Missing"] <- NA

  if (has_cat) {
    dt_levels <- c("<=", "<", "==")
    y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = dt_levels)
    y[is.na(Feature), Decision.type := NA]
    if (any(is_cat_split)) {
      data.table::set(y, which(is_cat_split), "Decision.type", "==")
    }
  } else {
    y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
    y[is.na(Feature), Decision.type := NA]
  }

  y <- y[, c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "ErrorReduction", "Cover")]
  colnames(y) <- c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover")

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y$Missing <- match(paste0(y$Missing, "-", y$Tree), ID)

  # Here we lose "Quality" information
  y[!is.na(Feature), Prediction := NA]

  # GBM calculates prediction as [initF + sum of predictions of trees]
  # treeSHAP assumes prediction are calculated as [sum of predictions of trees]
  # so here we adjust it
  ntrees <- sum(y$Node == 0)
  y[is.na(Feature), Prediction := Prediction + gbm_model$initF / ntrees]

  feature_names <- gbm_model$var.names
  data <- data[,colnames(data) %in% feature_names]

  ret <- list(model = as.data.frame(y), data = as.data.frame(data), feature_names = feature_names)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "gbm"

  # Original covers in gbm_model are not correct
  ret <- set_reference_dataset(ret, as.data.frame(data))

  return(ret)
}
