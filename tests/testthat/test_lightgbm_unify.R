library(treeshap)
param_lightgbm <- list(objective = "regression",
                       max_depth = 3,
                       force_row_wise = TRUE,
                       learning.rate = 0.1)

data_fifa <- fifa20$data[!colnames(fifa20$data)%in%c('work_rate', 'value_eur', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
data <- as.matrix(na.omit(data.table::as.data.table(cbind(data_fifa, fifa20$target))))
sparse_data <- data[,-ncol(data)]
x <- lightgbm::lgb.Dataset(sparse_data, label = data[,ncol(data)])
lgb_data <- lightgbm::lgb.Dataset.construct(x)
lgbm_fifa <- lightgbm::lightgbm(data = lgb_data,
                      params = param_lightgbm, verbose = -1,
                      save_name = paste0(tempfile(), '.model'))

lgbmtree <- lightgbm::lgb.model.dt.tree(lgbm_fifa)

test_that('lightgbm.unify returns an object with correct attributes', {
  unified_model <- lightgbm.unify(lgbm_fifa, sparse_data)

  expect_equal(attr(unified_model, "missing_support"), TRUE)
  expect_equal(attr(unified_model, "model"), "LightGBM")
})

test_that('Columns after lightgbm.unify are of appropriate type', {
  unified_model <- lightgbm.unify(lgbm_fifa, sparse_data)$model
  expect_true(is.integer(unified_model$Tree))
  expect_true(is.integer(unified_model$Node))
  expect_true(is.character(unified_model$Feature))
  expect_true(is.numeric(unified_model$Split))
  expect_true(is.integer(unified_model$Yes))
  expect_true(is.integer(unified_model$No))
  expect_true(is.integer(unified_model$Missing))
  expect_true(is.numeric(unified_model$Prediction))
  expect_true(is.numeric(unified_model$Cover))
})

test_that('lightgbm.unify creates an object of the appropriate class', {
  unified_model <- lightgbm.unify(lgbm_fifa, sparse_data)$model
  expect_true('data.table' %in% class(unified_model))
  expect_true('data.frame' %in% class(unified_model))
})

test_that('basic columns after lightgbm.unify are correct', {
  unified_model <- lightgbm.unify(lgbm_fifa, sparse_data)$model
  expect_equal(lgbmtree$tree_index, unified_model$Tree)
  to_test_features <- lgbmtree[order(lgbmtree$split_index), .(split_feature,split_index, threshold, leaf_count, internal_count),tree_index]
  expect_equal(to_test_features[!is.na(to_test_features$split_index),][['split_index']], unified_model[!is.na(Feature),][['Node']])
  expect_equal(to_test_features[['split_feature']], unified_model[['Feature']])
  expect_equal(to_test_features[['threshold']], unified_model[['Split']])
  expect_equal(to_test_features[!is.na(internal_count),][['internal_count']], unified_model[!is.na(Feature),][['Cover']])
})

test_that('connections between nodes and leaves after lightgbm.unify are correct', {
  test_object <- lightgbm.unify(lgbm_fifa, sparse_data)$model
  #Check if the sums of children's covers are correct
  expect_equal(test_object[test_object[!is.na(test_object$Yes)][['Yes']]][['Cover']] +
    test_object[test_object[!is.na(test_object$No)][['No']]][['Cover']], test_object[!is.na(Feature)][['Cover']])
  #check if default_left information is correctly used
  df_default_left <- lgbmtree[default_left == "TRUE", c('tree_index', 'split_index')]
  test_object_actual_default_left <- test_object[Yes == Missing, c('Tree', 'Node')]
  colnames(test_object_actual_default_left) <- c('tree_index', 'split_index')
  attr(test_object_actual_default_left, 'model') <- NULL
  expect_equal(test_object_actual_default_left[order(tree_index, split_index)], df_default_left[order(tree_index, split_index)])
  #and default_left = FALSE analogically:
  df_default_right <- lgbmtree[default_left != 'TRUE', c('tree_index', 'split_index')]
  test_object_actual_default_right <- test_object[No == Missing, c('Tree', 'Node')]
  colnames(test_object_actual_default_right) <- c('tree_index', 'split_index')
  attr(test_object_actual_default_right, 'model') <- NULL
  expect_equal(test_object_actual_default_right[order(tree_index, split_index)], df_default_right[order(tree_index, split_index)])
  #One more test with checking the usage of 'decision_type' column needed
})

# Function that return the predictions for sample observations indicated by vector contatining values -1, 0, 1, where -1 means
# going to the 'Yes' Node, 1 - to the 'No' node and 0 - to the missing node. The vectors are randomly produced during executing
# the function and should be passed to prepare_original_preds_ to save the conscistence. Later we can compare the 'predicted' values
prepare_test_preds <- function(unify_out){
  stopifnot(all(c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Prediction", "Cover") %in% colnames(unify_out)))
  test_tree <- unify_out[unify_out$Tree %in% 0:9,]
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
      #if(length(is.na(tree$Feature[indx]))>1) {print(paste(indx, i)); print(tree); print(obs)}
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

prepare_original_preds_lgbm <- function(orig_tree, test_obs){
  test_tree <- orig_tree[orig_tree$tree_index %in% 0:9,]
  test_tree <- split(test_tree, test_tree$tree_index)
  stopifnot(length(test_tree) == length(test_obs))
  determine_val <- function(obs, tree){
    i <- 1
    indx <- 1
    while(!is.na(tree$split_feature[indx])) {
      children <- ifelse(is.na(tree$node_parent), tree$leaf_parent, tree$node_parent)
      if((obs[i] < 0) | (tree$default_left[indx] == 'TRUE' & obs[i] == 0)){
        indx <- which(tree$split_index[indx] == children)[1]
      }
      else if((obs[i] > 0) | (tree$default_left[indx] == 'FALSE' & obs[i] == 0)) {
        indx <- which(tree$split_index[indx] == children)[2]
      }
      else{
        stop('Error in the connections')
        indx <- 0
      }
      i <- i + 1
    }
    return(tree[['leaf_value']][indx])
  }
  y = numeric()
  for(i in seq_along(test_obs)) {
    y[i] <- determine_val(test_obs[[i]], test_tree[[i]])
  }
  return(y)
}

test_that('the connections between the nodes are correct', {
  # The test is passed only if the predictions for sample observations are equal in the first 10 trees of the ensemble
  x <- prepare_test_preds(lightgbm.unify(lgbm_fifa, sparse_data)$model)
  preds <- x[['preds']]
  test_obs <- x[['test_obs']]
  original_preds <- prepare_original_preds_lgbm(lgbmtree, test_obs)
  expect_equal(preds, original_preds)
})
