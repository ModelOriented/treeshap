#' Compute Cover values of your model for reference dataset other than the original set used to train model.
#'
#'
#' @param model Unified dataframe representation of the model created with a (model).unify function.
#' @param X Reference dataset. A dataframe with the same columns as in the training set of the model.
#'
#'
#' @return  Unified dataframe representation of the model as created with a (model).unify function,
#' but with Cover column containing updated values.
#'
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
#' \dontrun{
#' library(gbm)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data['value_eur'] <- fifa20$target
#' gbm_model <- gbm::gbm(
#'              formula = value_eur ~ .,
#'              data = data,
#'              distribution = "laplace",
#'              n.trees = 1000,
#'              cv.folds = 2,
#'              interaction.depth = 2)
#' unified <- gbm.unify(gbm_model)
#' recalculate_covers(unified, data[200:700, ])
#'}
recalculate_covers <- function(model, X) {
  # argument check
  stopifnot(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover") %in% colnames(model))

  # functions wrapping tree structure
  is_leaf <- function(model, j) (is.na(model$Feature[j]))
  feature <- function(model, j) (model$Feature[j])
  lesser <- function(model, j) (model$Yes[j])
  greater <- function(model, j) (model$No[j])
  missing <- function(model, j) (model$Missing[j])
  threshold <- function(model, j) (model$Split[j])

  # recursively walk through a tree and update covers table
  rec_update_covers <- function(covers, model, X, passing, j) {
    covers[j] <- covers[j] + sum(passing & !is.na(passing))
    if (!is_leaf(model, j)) {
      condition <- X[[feature(model, j)]] <= threshold(model, j)
      #print(feature(model, j))
      #print(X[[feature(model, j)]])

      covers <- rec_update_covers(covers, model, X, is.na(condition) & passing, missing(model, j))
      covers <- rec_update_covers(covers, model, X, condition & passing, lesser(model, j))
      covers <- rec_update_covers(covers, model, X, !condition & passing, greater(model, j))
    }
    return(covers)
  }

  iterate_trees <- function(covers, model, X, roots) {
    if (length(roots) == 0) {
      covers
    } else {
      covers <- rec_update_covers(covers, model, X, rep(TRUE, nrow(X)), roots[1])
      iterate_trees(covers, model, X, roots[-1])
    }
  }

  # iterate over all observations and all trees
  roots <- which(model$Node == 0)
  covers <- rep(0, nrow(model))
  covers <- iterate_trees(covers, model, X, roots)

  model$Cover <- covers
  return(model)
}

