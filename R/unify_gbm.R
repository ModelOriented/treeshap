#' Unify GBM model
#'
#' Convert your GBM model into a standarised representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param gbm_model An object of \code{gbm} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{catboost.unify}} for  \code{\link[catboost:catboost.train]{Catboost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
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
  if(!inherits(gbm_model,'gbm')) {
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
  y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
  y[is.na(Feature), Decision.type := NA]
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

  ret <- list(model = as.data.frame(y), data = as.data.frame(data))
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "gbm"

  # Original covers in gbm_model are not correct
  ret <- set_reference_dataset(ret, as.data.frame(data))

  return(ret)
}
