library(treeshap)
library(xgboost)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
target <- fifa20$target
param <- list(objective = "reg:squarederror", max_depth = 3)
xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200, verbose = 0)
xgb_tree <- xgboost::xgb.model.dt.tree(model = xgb_model)


test_that('xgboost.unify returns an object of appropriate class', {
  expect_true('data.table' %in% class(xgboost.unify(xgb_model, as.matrix(data))$model))
  expect_true('data.frame' %in% class(xgboost.unify(xgb_model, as.matrix(data))$model))
})

test_that('columns after xgboost.unify are of appropriate type', {
  expect_true(is.integer(xgboost.unify(xgb_model, as.matrix(data))$model$Tree))
  expect_true(is.integer(xgboost.unify(xgb_model, as.matrix(data))$model$Node))
  expect_true(is.character(xgboost.unify(xgb_model, as.matrix(data))$model$Feature))
  expect_true(is.numeric(xgboost.unify(xgb_model, as.matrix(data))$model$Split))
  expect_true(is.integer(xgboost.unify(xgb_model, as.matrix(data))$model$Yes))
  expect_true(is.integer(xgboost.unify(xgb_model, as.matrix(data))$model$No))
  expect_true(is.integer(xgboost.unify(xgb_model, as.matrix(data))$model$Missing))
  expect_true(is.numeric(xgboost.unify(xgb_model, as.matrix(data))$model[['Quality/Score']]))
  expect_true(is.numeric(xgboost.unify(xgb_model, as.matrix(data))$model$Cover))
})

test_that('values in the columns after xgboost.unify are correct', {
  expect_equal(xgb_tree$Tree, xgboost.unify(xgb_model, as.matrix(data))$model$Tree)
  expect_equal(xgb_tree$Node, xgboost.unify(xgb_model, as.matrix(data))$model$Node)
  expect_equal(xgb_tree$Cover, xgboost.unify(xgb_model, as.matrix(data))$model$Cover)
  expect_equal(xgb_tree$Quality, xgboost.unify(xgb_model, as.matrix(data))$model[['Quality/Score']])
  expect_equal(xgb_tree$Node, xgboost.unify(xgb_model, as.matrix(data))$model$Node)
  expect_equal(xgb_tree$Split, xgboost.unify(xgb_model, as.matrix(data))$model$Split)
  expect_equal(match(xgb_tree$Yes, xgb_tree$ID), xgboost.unify(xgb_model, as.matrix(data))$model$Yes)
  expect_equal(match(xgb_tree$No, xgb_tree$ID), xgboost.unify(xgb_model, as.matrix(data))$model$No)
  expect_equal(match(xgb_tree$Missing, xgb_tree$ID), xgboost.unify(xgb_model, as.matrix(data))$model$Missing)
  expect_equal(xgb_tree[xgb_tree[['Feature']] != 'Leaf',][['Feature']],
               xgboost.unify(xgb_model, as.matrix(data))$model[!is.na(xgboost.unify(xgb_model, as.matrix(data))$model$Feature),][['Feature']])
  expect_equal(nrow(xgb_tree[xgb_tree[['Feature']] == 'Leaf',]),
               nrow(xgboost.unify(xgb_model, as.matrix(data))$model[is.na(xgboost.unify(xgb_model, as.matrix(data))$model$Feature),]))

})



test_that('xgboost.unify() does not work for objects produced with other packages', {
  param_lightgbm <- list(objective = "regression",
                         max_depth = 2,
                         num_leaves = 4L,
                         force_row_wise = TRUE,
                         learning.rate = 0.1)
  expect_warning({lgbm_fifa <- lightgbm::lightgbm(data = as.matrix(fifa20$data[colnames(fifa20$data) != 'value_eur']),
                        label = fifa20$target,
                        params = param_lightgbm,
                        verbose = -1,
                        save_name = paste0(tempfile(), '.model'))
  lgbmtree <- lightgbm::lgb.model.dt.tree(lgbm_fifa)})
  expect_error(xgboost.unify(lgbmtree))
})

# Function that return the predictions for sample observations indicated by vector contatining values -1, 0, 1, where -1 means
# going to the 'Yes' Node, 1 - to the 'No' node and 0 - to the missing node. The vectors are randomly produced during executing
# the function and should be passed to prepare_original_preds_ to save the conscistence. Later we can compare the 'predicted' values
prepare_test_preds <- function(unify_out){
  stopifnot(all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover") %in% colnames(unify_out)))
  test_tree <- unify_out[unify_out$Tree %in% 0:9, ]
  test_tree[['node_row_id']] <- seq_len(nrow(test_tree))
  test_obs <- lapply(table(test_tree$Tree), function(y) sample(c(-1, 0, 1), y, replace = T))
  test_tree <- split(test_tree, test_tree$Tree)
  determine_val <- function(obs, tree){
    root_id <- tree[['node_row_id']][1]
    tree[,c('Yes', 'No', 'Missing')] <- tree[,c('Yes', 'No', 'Missing')] - root_id + 1
    i <- 1
    indx <- 1
    while(!is.na(tree$Feature[indx])) {
      indx <- ifelse(obs[i] == 0, tree$Missing[indx], ifelse(obs[i] < 0, tree$Yes[indx], tree$No[indx]))
      i <- i + 1
    }
    return(tree[['Quality/Score']][indx])
  }
  x = numeric()
  for(i in seq_along(test_obs)) {
    x[i] <- determine_val(test_obs[[i]], test_tree[[i]])

  }
  return(list(preds = x, test_obs = test_obs))
}

prepare_original_preds_xgb <- function(orig_tree, test_obs){
  test_tree <- orig_tree[orig_tree$Tree %in% 0:9, ]
  test_tree <- split(test_tree, test_tree$Tree)
  stopifnot(length(test_tree) == length(test_obs))
  determine_val <- function(obs, tree){
    i <- 1
    indx <- 1
    while(!is.na(tree$Split[indx])) {
      indx <- ifelse(obs[i] == 0, match(tree$Missing[indx], tree$ID), ifelse(obs[i] < 0, match(tree$Yes[indx], tree$ID),
                                                                             match(tree$No[indx], tree$ID)))

      i <- i + 1
    }
    return(tree[['Quality']][indx])
  }
  y = numeric()
  for(i in seq_along(test_obs)) {
    y[i] <- determine_val(test_obs[[i]], test_tree[[i]])
  }
  return(y)
}

test_that('the connections between the nodes are correct', {
  # The test is passed only if the predictions for sample observations are equal in the first 10 trees of the ensemble
  x <- prepare_test_preds(xgboost.unify(xgb_model, as.matrix(data))$model)
  preds <- x[['preds']]
  test_obs <- x[['test_obs']]
  original_preds <- prepare_original_preds_xgb(xgb_tree, test_obs)
  expect_equal(preds, original_preds)
})
