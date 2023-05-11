#' Unify randomForest model
#'
#' Convert your randomForest model into a standarised representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param rf_model An object of \code{randomForest} class. At the moment, models built on data with categorical features
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
#' \code{\link{catboost.unify}} for  \code{\link[catboost:catboost.train]{Catboost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
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
  if(!inherits(rf_model,'randomForest')){stop('Object rf_model was not of class "randomForest"')}
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
  setnames(y, c("Yes", "No", "Feature", "Split",  "Prediction", "Tree", "Node"))
  y[, Feature := as.character(Feature)]
  y[, Yes := Yes - 1]
  y[, No := No - 1]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  y[, Missing := NA]
  y[, Missing := as.integer(Missing)] # seems not, but needed

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)

  y$Cover <- 0

  y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
  y[is.na(Feature), Decision.type := NA]

  # Here we lose "Quality" information
  y[!is.na(Feature), Prediction := NA]

  # treeSHAP assumes, that [prediction = sum of predictions of the trees]
  # in random forest [prediction = mean of predictions of the trees]
  # so here we correct it by adjusting leaf prediction values
  y[is.na(Feature), Prediction := Prediction / n]


  setcolorder(y, c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover"))

  ret <- list(model = as.data.frame(y), data = as.data.frame(data))
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- FALSE
  attr(ret, "model") <- "randomForest"
  return(set_reference_dataset(ret, as.data.frame(data)))
}
