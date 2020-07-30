library(treeshap)
library(catboost)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
label <- fifa20$target
dt.pool <- catboost::catboost.load_pool(data = as.data.frame(lapply(data, as.numeric)), label = label)
cat_model <- catboost::catboost.train(
            dt.pool,
            params = list(loss_function = 'RMSE',
                          iterations = 100,
                          metric_period = 10,
                          logging_level = 'Silent',
                        allow_writing_files = FALSE))
catboost.unify(cat_model, dt.pool)

test_that('catboost.unify returns an object of appropriate class', {
 expect_true('data.table' %in% class(catboost.unify(cat_model, dt.pool)))
 expect_true('data.frame' %in% class(catboost.unify(cat_model, dt.pool)))
})


test_that('catboost raises an appropriate error when a model with categorical variables is used', {
  data['work_rate'] <- fifa20$data[,'work_rate']
  dt.pool_cat <- catboost::catboost.load_pool(data = data, label = label)
  cat_model_cat <- catboost::catboost.train(dt.pool_cat,
                  params = list(loss_function = 'RMSE',
                  iterations = 100,
                  metric_period = 10,
                  logging_level = 'Silent',
                  allow_writing_files = FALSE))
  expect_error(catboost.unify(cat_model_cat, dt.pool_cat))
})


test_that('columns after catboost.unify are of appropriate type', {
 expect_true(is.integer(catboost.unify(cat_model, dt.pool)$Tree))
 expect_true(is.integer(catboost.unify(cat_model, dt.pool)$Node))
 expect_true(is.character(catboost.unify(cat_model, dt.pool)$Feature))
 expect_true(is.numeric(catboost.unify(cat_model, dt.pool)$Split))
 expect_true(is.integer(catboost.unify(cat_model, dt.pool)$Yes))
 expect_true(is.integer(catboost.unify(cat_model, dt.pool)$No))
 expect_true(is.integer(catboost.unify(cat_model, dt.pool)$Missing))
 expect_true(is.numeric(catboost.unify(cat_model, dt.pool)[['Quality/Score']]))
 expect_true(is.numeric(catboost.unify(cat_model, dt.pool)$Cover))
})

path_to_save <- paste0(tempdir(), '/catboostmodel_test.json')
catboost::catboost.save_model(cat_model, path_to_save, 'json')
json_data <- jsonlite::read_json(path_to_save)

test_that('values in the Split column after catboost.unify are correct', {
  indx <- seq_along(json_data$oblivious_trees)
  borders <- lapply(json_data$oblivious_trees[indx], function(x) sapply(x$splits, function(y) y$border))
  repeated_borders <- lapply(borders, function(x) x[rep(seq_along(x), times = 2^(seq_along(x)-1))])
  to_test_splits <- do.call(c, repeated_borders)
  expect_equal(to_test_splits,  catboost.unify(cat_model, dt.pool)[(Tree %in% (indx-1)) & !is.na(Split)][['Split']])
})


# Function that return the predictions for sample observations indicated by vector contatining values -1, 0, 1, where -1 means
# going to the 'Yes' Node, 1 - to the 'No' node and 0 - to the missing node. The vectors are randomly produced during executing
# the function and should be passed to prepare_original_preds_ to save the conscistence. Later we can compare the 'predicted' values
# For the purpose of testing catboost.unify, all zeros prodcued in the samples are treated as -1. NAs will be tested later
prepare_test_preds <- function(unify_out){
  stopifnot(all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover") %in% colnames(unify_out)))
  test_tree <- unify_out[unify_out$Tree %in% 0:19,]
  test_tree[['node_row_id']] <- seq_len(nrow(test_tree))
  test_obs <- lapply(table(test_tree$Tree), function(y) sample(c(-1, 0, 1), floor(log(y, 2)), replace = T))
  test_tree <- split(test_tree, test_tree$Tree)
  determine_val <- function(obs, tree){
    root_id <- tree[['node_row_id']][1]
    tree[,c('Yes', 'No', 'Missing')] <- tree[,c('Yes', 'No', 'Missing')] - root_id + 1
    i <- 1
    indx <- 1
    while(!is.na(tree$Feature[indx])) {
      indx <- ifelse(obs[i] == 0, ifelse(is.na(tree$Missing[indx]), tree$Yes[indx], tree$Missing[indx]),
                     ifelse(obs[i]<0, tree$Yes[indx], tree$No[indx])) #if the value of Missing is NA - take the Yes node
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


prepare_original_preds_catboost <- function(orig_json_data, test_obs){

  determine_val <- function(obs, orig_json_data, i){
    features <- sapply(orig_json_data$oblivious_trees[[i]]$splits, function(x) x$float_feature_index)
    na_treatments <- sapply(features, function(x) orig_json_data$features_info$float_features[[x+1]]$nan_value_treatment)
    stopifnot(length(obs) == length(features))
    obs[obs == 0 & na_treatments %in% c('AsIs', 'AsFalse')] <- -1
    obs[obs == 0 & na_treatments == 'AsTrue'] <- 1
    obs[obs == -1] <- 0
    values_vector <- unlist(orig_json_data$oblivious_trees[[i]]$leaf_values)
    stopifnot(length(values_vector) == 2^length(obs))
    value_indx <- strtoi(paste0(obs, collapse = ""), 2)+1
    #print(value_indx)
    values_vector[value_indx]
  }

  y = numeric()
  for(i in seq_along(test_obs)) {
    test_obs[[i]][test_obs[[i]] == -1] <- 0
    y[i] <- determine_val(test_obs[[i]], orig_json_data, i)
  }
  return(y)
}

test_that('connections between nodes after catboost.unify are correct',{
  x <- prepare_test_preds(catboost.unify(cat_model , dt.pool))
  test_obs <- x[['test_obs']]
  preds <- x[['preds']]
  original_preds <- prepare_original_preds_catboost(json_data, test_obs)
  expect_equal(preds, original_preds)
})

