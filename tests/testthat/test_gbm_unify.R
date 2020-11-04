library(gbm)
library(treeshap)
x <- fifa20$data[colnames(fifa20$data) != 'work_rate']
x['value_eur'] <- fifa20$target

chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")

if (nzchar(chk) && chk == "TRUE") {
  # use 2 cores in CRAN/Travis/AppVeyor
  num_workers <- 2L
} else {
  # use all cores in devtools::test()
  num_workers <- parallel::detectCores()
}

gbm_num_model <- gbm::gbm(
  formula = value_eur ~ .,
  data = x,
  n.trees = 100,
  distribution = 'gaussian',
  cv.folds = 2,
  n.cores = num_workers
)

x['work_rate'] <- fifa20$data['work_rate']
gbm_with_cat_model <- gbm::gbm(
  formula = value_eur ~ .,
  data = x,
  distribution = "laplace",
  n.trees = 100,
  cv.folds = 2,
  interaction.depth = 2,
  n.cores = num_workers
)

test_that('the gbm.unify function does not support models with categorical features', {
  expect_error(gbm.unify(gbm_with_cat_model), "Models built on data with categorical features are not supported - please encode them before training.")
})

test_that('the gbm.unify function returns data frame with columns of appropriate column', {
  expect_true(is.integer(gbm.unify(gbm_num_model)$Tree))
  expect_true(is.integer(gbm.unify(gbm_num_model)$Node))
  expect_true(is.character(gbm.unify(gbm_num_model)$Feature))
  expect_true(is.numeric(gbm.unify(gbm_num_model)$Split))
  expect_true(is.integer(gbm.unify(gbm_num_model)$Yes))
  expect_true(is.integer(gbm.unify(gbm_num_model)$No))
  expect_true(is.integer(gbm.unify(gbm_num_model)$Missing))
  expect_true(is.numeric(gbm.unify(gbm_num_model)[['Quality/Score']]))
  expect_true(is.numeric(gbm.unify(gbm_num_model)$Cover))
})

gbm_tree <- pretty.gbm.tree(gbm_num_model, 1)
gbm_tree['tree_index'] <- 0
gbm_tree['node_index']  <- seq_len(nrow(gbm_tree)) - 1
for (i in 2:gbm_num_model$n.trees){
  new <- pretty.gbm.tree(gbm_num_model, i)
  new['tree_index'] <- i - 1
  new['node_index']  <- seq_len(nrow(new)) - 1
  gbm_tree <- rbind(gbm_tree, new)
}
rownames(gbm_tree) <- NULL

test_that('basic columns after gbm.unify are correct', {
  expect_equal(gbm_num_model$var.names[gbm_tree$SplitVar + 1], gbm.unify(gbm_num_model)[!is.na(Feature)]$Feature)
  expect_equal(gbm_tree[gbm_tree$SplitVar < 0, ][['Prediction']], gbm.unify(gbm_num_model)[is.na(Feature)][['Quality/Score']])
  expect_equal(gbm_tree[gbm_tree$SplitVar >= 0, ][['ErrorReduction']], gbm.unify(gbm_num_model)[!is.na(Feature)][['Quality/Score']])
  expect_equal(gbm_tree$Weight, gbm.unify(gbm_num_model)$Cover)
  expect_equal(match(paste0(gbm_tree$tree_index, '-', gbm_tree$LeftNode), paste0(gbm_tree$tree_index, '-', gbm_tree$node_index)),
               gbm.unify(gbm_num_model)$Yes)
  expect_equal(match(paste0(gbm_tree$tree_index, '-', gbm_tree$RightNode), paste0(gbm_tree$tree_index, '-', gbm_tree$node_index)),
               gbm.unify(gbm_num_model)$No)
  expect_equal(match(paste0(gbm_tree$tree_index, '-', gbm_tree$MissingNode), paste0(gbm_tree$tree_index, '-', gbm_tree$node_index)),
               gbm.unify(gbm_num_model)$Missing)
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
      #if(length(is.na(tree$Feature[indx]))>1) {print(paste(indx, i)); print(tree); print(obs)}
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


prepare_original_preds_gbm <- function(orig_tree, test_obs){
  test_tree <- orig_tree[orig_tree$tree_index %in% 0:9, ]
  test_tree <- split(test_tree, test_tree$tree_index)
  stopifnot(length(test_tree) == length(test_obs))
  determine_val <- function(obs, tree){
    i <- 1
    indx <- 1
    while(tree$SplitVar[indx]>=0) {
      indx <- ifelse(obs[i] == 0, tree$MissingNode[indx], ifelse(obs[i] < 0, tree$LeftNode[indx], tree$RightNode[indx])) + 1 # +1 caused by 0-indexing
      i <- i + 1
    }
    return(tree[['Prediction']][indx])
  }
  y = numeric()
  for(i in seq_along(test_obs)) {
    y[i] <- determine_val(test_obs[[i]], test_tree[[i]])
  }
  return(y)
}

test_that('the connections between the nodes are correct', {
  # The test is passed only if the predictions for sample observations are equal in the first 10 trees of the ensemble
  x <- prepare_test_preds(gbm.unify(gbm_num_model))
  preds <- x[['preds']]
  test_obs <- x[['test_obs']]
  original_preds <- prepare_original_preds_gbm(gbm_tree, test_obs)
  expect_equal(preds, original_preds)
})
