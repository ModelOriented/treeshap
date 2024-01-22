library(treeshap)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
target <- fifa20$target
param <- list(objective = "reg:squarederror", max_depth = 3)
xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200, verbose = 0)
xgb_tree <- xgboost::xgb.model.dt.tree(model = xgb_model)


test_that('xgboost.unify returns an object of appropriate class', {
  expect_true(is.model_unified(xgboost.unify(xgb_model, as.matrix(data))))
  expect_true(is.model_unified(unify(xgb_model, as.matrix(data))))
})

test_that('xgboost.unify returns an object with correct attributes', {
  unified_model <- xgboost.unify(xgb_model, as.matrix(data))

  expect_equal(attr(unified_model, "missing_support"), TRUE)
  expect_equal(attr(unified_model, "model"), "xgboost")
})

test_that('columns after xgboost.unify are of appropriate type', {
  unified_model <- xgboost.unify(xgb_model, as.matrix(data))$model

  expect_true(is.integer(unified_model$Tree))
  expect_true(is.integer(unified_model$Node))
  expect_true(is.character(unified_model$Feature))
  expect_true(is.factor(unified_model$Decision.type))
  expect_true(is.numeric(unified_model$Split))
  expect_true(is.integer(unified_model$Yes))
  expect_true(is.integer(unified_model$No))
  expect_true(is.integer(unified_model$Missing))
  expect_true(is.numeric(unified_model$Prediction))
  expect_true(is.numeric(unified_model$Cover))
})

test_that('values in the columns after xgboost.unify are correct', {
  unified_model <- xgboost.unify(xgb_model, as.matrix(data))$model

  expect_equal(xgb_tree$Tree, unified_model$Tree)
  expect_equal(xgb_tree$Node, unified_model$Node)
  expect_equal(xgb_tree$Cover, unified_model$Cover)
  expect_equal(xgb_tree$Quality[xgb_tree$Feature == 'Leaf'], unified_model$Prediction[is.na(unified_model$Feature)])
  expect_equal(xgb_tree$Node, unified_model$Node)
  expect_equal(xgb_tree$Split, unified_model$Split)
  expect_equal(match(xgb_tree$Yes, xgb_tree$ID), unified_model$Yes)
  expect_equal(match(xgb_tree$No, xgb_tree$ID), unified_model$No)
  expect_equal(match(xgb_tree$Missing, xgb_tree$ID), unified_model$Missing)
  expect_equal(xgb_tree[xgb_tree[['Feature']] != 'Leaf',][['Feature']],
               unified_model[!is.na(unified_model$Feature),][['Feature']])
  expect_equal(nrow(xgb_tree[xgb_tree[['Feature']] == 'Leaf',]),
               nrow(unified_model[is.na(unified_model$Feature),]))

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
                        num_threads = 0)
  lgbmtree <- lightgbm::lgb.model.dt.tree(lgbm_fifa)})
  expect_error(xgboost.unify(lgbmtree))
})

# Function that return the predictions for sample observations indicated by vector contatining values -1, 0, 1, where -1 means
# going to the 'Yes' Node, 1 - to the 'No' node and 0 - to the missing node. The vectors are randomly produced during executing
# the function and should be passed to prepare_original_preds_ to save the conscistence. Later we can compare the 'predicted' values
prepare_test_preds <- function(unify_out){
  stopifnot(all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Prediction", "Cover") %in% colnames(unify_out)))
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
    return(tree[['Prediction']][indx])
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

test_that("xgboost: predictions from unified == original predictions", {
  unifier <- xgboost.unify(xgb_model, data)
  obs <- data[1:16000, ]
  original <- stats::predict(xgb_model, as.matrix(obs))
  from_unified <- predict(unifier, obs)
  # expect_equal(from_unified, original) #there are small differences
  expect_true(all(abs((from_unified - original) / original) < 5 * 10**(-3)))
})

test_that("xgboost: mean prediction calculated using predict == using covers", {
  unifier <- xgboost.unify(xgb_model, data)

  intercept_predict <- mean(predict(unifier, data))

  ntrees <- sum(unifier$model$Node == 0)
  leaves <- unifier$model[is.na(unifier$model$Feature), ]
  intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

  #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
  expect_equal(intercept_predict, intercept_covers)
})

test_that("xgboost: covers correctness", {
  unifier <- xgboost.unify(xgb_model, data)

  roots <- unifier$model[unifier$model$Node == 0, ]
  expect_true(all(roots$Cover == nrow(data)))

  internals <- unifier$model[!is.na(unifier$model$Feature), ]
  yes_child_cover <- unifier$model[internals$Yes, ]$Cover
  no_child_cover <- unifier$model[internals$No, ]$Cover
  if (all(is.na(internals$Missing))) {
    children_cover <- yes_child_cover + no_child_cover
  } else {
    missing_child_cover <- unifier$model[internals$Missing, ]$Cover
    missing_child_cover[is.na(missing_child_cover)] <- 0
    missing_child_cover[internals$Missing == internals$Yes | internals$Missing == internals$No] <- 0
    children_cover <- yes_child_cover + no_child_cover + missing_child_cover
  }
  expect_true(all(internals$Cover == children_cover))
})
