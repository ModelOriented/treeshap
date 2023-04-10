library(treeshap)

skip_if_no_catboost <- function(){
  if (!requireNamespace("catboost", quietly = TRUE)) {
    skip("catboost not installed")
  }
}

data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
data <- as.data.frame(lapply(data, as.numeric))
label <- fifa20$target



test_that('catboost.unify returns an object of appropriate class', {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))
  expect_true(is.model_unified(catboost.unify(cat_model, data)))
})

test_that('catboost.unify returns an object with correct attributes', {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  unified_model <- catboost.unify(cat_model, data)

  expect_equal(attr(unified_model, "missing_support"), TRUE)
  expect_equal(attr(unified_model, "model"), "catboost")
})

test_that('catboost raises an appropriate error when a model with categorical variables is used', {
  skip_if_no_catboost()

  data['work_rate'] <- fifa20$data[,'work_rate']
  dt.pool_cat <- catboost::catboost.load_pool(data = data, label = label)
  cat_model_cat <- catboost::catboost.train(dt.pool_cat,
                  params = list(loss_function = 'RMSE',
                  iterations = 100,
                  metric_period = 10,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  expect_error(catboost.unify(cat_model_cat, data))
})


test_that('columns after catboost.unify are of appropriate type', {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  model <- catboost.unify(cat_model, data)$model
  expect_true(is.integer(model$Tree))
  expect_true(is.integer(model$Node))
  expect_true(is.character(model$Feature))
  expect_true(is.factor(model$Decision.type))
  expect_true(is.numeric(model$Split))
  expect_true(is.integer(model$Yes))
  expect_true(is.integer(model$No))
  expect_true(is.integer(model$Missing))
  expect_true(is.numeric(model$Prediction))
  expect_true(is.numeric(model$Cover))
})

test_that("catboost: predictions from unified == original predictions", {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  unifier <- catboost.unify(cat_model, data)
  obs <- c(1:16000)
  original <- catboost::catboost.predict(cat_model, catboost::catboost.load_pool(data = data[obs, ], label = label[obs]))
  from_unified <- predict(unifier, data[obs, ])
  # expect_equal(from_unified, original) #there are small differences
  expect_true(all(abs((from_unified - original) / original) < 10**(-11)))
})

test_that("catboost: mean prediction calculated using predict == using covers", {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  unifier <- catboost.unify(cat_model, data)

  intercept_predict <- mean(predict(unifier, data))

  ntrees <- sum(unifier$model$Node == 0)
  leaves <- unifier$model[is.na(unifier$model$Feature), ]
  intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

  #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
  expect_equal(intercept_predict, intercept_covers)
})

test_that("catboost: covers correctness", {
  skip_if_no_catboost()

  dt.pool <- catboost::catboost.load_pool(data = data, label = label)
  cat_model <- catboost::catboost.train(
    dt.pool,
    params = list(loss_function = 'RMSE',
                  iterations = 100,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))

  unifier <- catboost.unify(cat_model, data)

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

# path_to_save <- paste0(tempdir(), '/catboostmodel_test.json')
# catboost::catboost.save_model(cat_model, path_to_save, 'json')
# json_data <- jsonlite::read_json(path_to_save)
#
# test_that('values in the Split column after catboost.unify are correct', {
#   indx <- seq_along(json_data$oblivious_trees)
#   borders <- lapply(json_data$oblivious_trees[indx], function(x) sapply(x$splits, function(y) y$border))
#   repeated_borders <- lapply(borders, function(x) x[rep(seq_along(x), times = 2^(seq_along(x)-1))])
#   to_test_splits <- do.call(c, repeated_borders)
#   expect_equal(to_test_splits,  catboost.unify(cat_model, dt.pool)[(Tree %in% (indx-1)) & !is.na(Split)][['Split']])
# })
#
#
# # Function that return the predictions for sample observations indicated by vector contatining values -1, 0, 1, where -1 means
# # going to the 'Yes' Node, 1 - to the 'No' node and 0 - to the missing node. The vectors are randomly produced during executing
# # the function and should be passed to prepare_original_preds_ to save the conscistence. Later we can compare the 'predicted' values
# # For the purpose of testing catboost.unify, all zeros prodcued in the samples are treated as -1. NAs will be tested later
# prepare_test_preds <- function(unify_out){
#   stopifnot(all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Prediction", "Cover") %in% colnames(unify_out)))
#   test_tree <- unify_out[unify_out$Tree %in% 0:19,]
#   test_tree[['node_row_id']] <- seq_len(nrow(test_tree))
#   test_obs <- lapply(table(test_tree$Tree), function(y) sample(c(-1, 0, 1), floor(log(y, 2)), replace = T))
#   test_tree <- split(test_tree, test_tree$Tree)
#   determine_val <- function(obs, tree){
#     root_id <- tree[['node_row_id']][1]
#     tree[,c('Yes', 'No', 'Missing')] <- tree[,c('Yes', 'No', 'Missing')] - root_id + 1
#     i <- 1
#     indx <- 1
#     while(!is.na(tree$Feature[indx])) {
#       indx <- ifelse(obs[i] == 0, ifelse(is.na(tree$Missing[indx]), tree$Yes[indx], tree$Missing[indx]),
#                      ifelse(obs[i]<0, tree$Yes[indx], tree$No[indx])) #if the value of Missing is NA - take the Yes node
#       i <- i + 1
#     }
#     return(tree[['Prediction']][indx])
#   }
#   x = numeric()
#   for(i in seq_along(test_obs)) {
#     x[i] <- determine_val(test_obs[[i]], test_tree[[i]])
#
#   }
#   return(list(preds = x, test_obs = test_obs))
# }
#
#
# prepare_original_preds_catboost <- function(orig_json_data, test_obs){
#
#   determine_val <- function(obs, orig_json_data, i){
#     features <- sapply(orig_json_data$oblivious_trees[[i]]$splits, function(x) x$float_feature_index)
#     na_treatments <- sapply(features, function(x) orig_json_data$features_info$float_features[[x+1]]$nan_value_treatment)
#     stopifnot(length(obs) == length(features))
#     obs[obs == 0 & na_treatments %in% c('AsIs', 'AsFalse')] <- -1
#     obs[obs == 0 & na_treatments == 'AsTrue'] <- 1
#     obs[obs == -1] <- 0
#     values_vector <- unlist(orig_json_data$oblivious_trees[[i]]$leaf_values)
#     stopifnot(length(values_vector) == 2^length(obs))
#     value_indx <- strtoi(paste0(obs, collapse = ""), 2)+1
#     #print(value_indx)
#     values_vector[value_indx]
#   }
#
#   y = numeric()
#   for(i in seq_along(test_obs)) {
#     test_obs[[i]][test_obs[[i]] == -1] <- 0
#     y[i] <- determine_val(test_obs[[i]], orig_json_data, i)
#   }
#   return(y)
# }
#
# test_that('connections between nodes after catboost.unify are correct',{
#   x <- prepare_test_preds(catboost.unify(cat_model , dt.pool))
#   test_obs <- x[['test_obs']]
#   preds <- x[['preds']]
#   original_preds <- prepare_original_preds_catboost(json_data, test_obs)
#   expect_equal(preds, original_preds)
# })
#

