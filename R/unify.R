#' Unify tree-based model
#'
#' Convert your tree-based model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param model A tree-based model object of any supported class (\code{gbm}, \code{lgb.Booster}, \code{randomForest}, \code{ranger}, \code{xgb.Booster}, or \code{catboost.Model}).
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#' @param ... Additional parameters passed to the model-specific unification functions.
#'
#' @return A unified model representation - a \code{\link{model_unified.object}} object
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{catboost.unify}} for \code{\link[catboost:catboost.train]{CatBoost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @export
#'
#' @examples
#'
#'  library(ranger)
#'  data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'                             c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'                              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#'  data <- na.omit(cbind(data_fifa, target = fifa20$target))
#'
#'  rf1 <- ranger::ranger(target~., data = data, max.depth = 10, num.trees = 10)
#'  unified_model1 <- unify(rf1, data)
#'  shaps1 <- treeshap(unified_model, data[1:2,])
#'  plot_contribution(shaps1, obs = 1)
#'
#'  rf2 <- randomForest::randomForest(target~., data = data, maxnodes = 10, ntree = 10)
#'  unified_model2 <- unify(rf2, data)
#'  shaps2 <- treeshap(unified_model2, data[1:2,])
#'  plot_contribution(shaps2, obs = 1)
unify <- function(model, data, ...){
  UseMethod("unify", model)
}

#' @export
unify.gbm <- function(model, data, ...){
  gbm.unify(model, data)
}

#' @export
unify.lgb.Booster <- function(model, data, recalculate = FALSE, ...){
  lightgbm.unify(model, data, recalculate)
}

#' @export
unify.randomForest <- function(model, data, ...){
  randomForest.unify(model, data)
}

#' @export
unify.ranger <- function(model, data, ...){
  if (model$treetype == "Survival"){
    return(ranger_surv.unify(model, data, ...))
  }
  ranger.unify(model, data)
}

#' @export
unify.xgb.Booster <- function(model, data, recalculate = FALSE, ...){
  xgboost.unify(model, data, recalculate)
}

#' @export
unify.catboost.Model <- function(model, data, recalculate = FALSE, ...){
  catboost.unify(model, data, recalculate)
}

#' @export
unify.default <- function(model, data, ...){
  stop("Provided model is not of type supported by treeshap.")
}

