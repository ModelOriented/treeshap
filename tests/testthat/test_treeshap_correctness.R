## Small tests comparing TreeSHAP results to brutal implementation results
## Extensive testing of correctness is not possible due to complexity of brutal implementation

library(treeshap)

skip_if_no_catboost <- function(){
  if (!requireNamespace("catboost", quietly = TRUE)) {
    skip("catboost not installed")
  }
}


data <- fifa20$data[, 3:6] # limiting columns for faster exponential calculation
stopifnot(all(!is.na(data)))

data_na <- fifa20$data[, c(3:5, 9)] # limiting columns for faster exponential calculation
stopifnot(any(is.na(data_na)))

target <- fifa20$target

test_model <- function(max_depth, nrounds, model = "xgboost",
                       test_data = data, test_target = target) {
  if (model == "xgboost") {
    param <- list(objective = "reg:squarederror", max_depth = max_depth)
    xgb_model <- xgboost::xgboost(as.matrix(test_data), params = param, label = target, nrounds = nrounds, verbose = FALSE)
    return(xgboost.unify(xgb_model, data))
  } else if (model == "ranger") {
    if (any(is.na(test_data))) stop("ranger does not work with NAa")
    rf <- ranger::ranger(test_target ~ ., data = test_data, max.depth = max_depth, num.trees = nrounds)
    return(ranger.unify(rf, test_data))
  } else if (model == "randomForest") {
    if (any(is.na(test_data))) stop("randomForest does not work with NAa")
    rf <- randomForest::randomForest(test_target ~ ., data = test_data, maxnodes = 2 ** max_depth, ntree = nrounds)
    return(randomForest.unify(rf, test_data))
  } else if (model == "gbm") {
    gbm_data <- test_data
    gbm_data["gbm_target"] <- test_target
    gbm_model <- gbm::gbm(
                 formula = gbm_target ~ .,
                 data = gbm_data,
                 distribution = "gaussian",
                 n.trees = nrounds,
                 interaction.depth = max_depth,
                 n.cores = 1)
    return(gbm.unify(gbm_model, gbm_data))
  } else if (model == "lightgbm") {
    param_lgbm <- list(objective = "regression", max_depth = max_depth, force_row_wise = TRUE)
    x <- lightgbm::lgb.Dataset(as.matrix(test_data), label = as.matrix(test_target))
    lgb_data <- lightgbm::lgb.Dataset.construct(x)
    lgb_model <- lightgbm::lightgbm(data = lgb_data, params = param_lgbm, nrounds = nrounds, verbose = -1,
                                    save_name = paste0(tempfile(), '.model'))
    return(lightgbm.unify(lgb_model, as.matrix(test_data)))
  } else if (model == "catboost") {

    skip_if_no_catboost()

    data <- as.data.frame(lapply(test_data, as.numeric))
    dt.pool <- catboost::catboost.load_pool(data = data, label = test_target)
    cat_model <- catboost::catboost.train(
      dt.pool,
      params = list(loss_function = 'RMSE',
                    iterations = nrounds,
                    depth = max_depth,
                    logging_level = 'Silent',
                    allow_writing_files = FALSE))
    return(catboost.unify(cat_model, data))
  }
}


## Implementation of exponential complexity SHAP calculation

# functions wrapping tree structure
is_leaf <- function(model, j) (is.na(model$Feature[j]))
leaf_value <- function(model, j) {
  stopifnot(is_leaf(model, j))
  model$Prediction[j]
}
feature <- function(model, j) (model$Feature[j])
lesser <- function(model, j) (model$Yes[j])
greater <- function(model, j) (model$No[j])
missing <- function(model, j) (model$Missing[j])
threshold <- function(model, j) (model$Split[j])
cover <- function(model, j) (model$Cover[j])
extract_tree_root <- function(model, i) (which((model$Tree == i) & (model$Node == 0)))
tree_ids <- function(model) (unique(model$Tree))

# function estimating E[f(x) | x_S]
expvalue <- function(tree, root, x, S) {
  n <- ncol(x)
  G <- function(j, w) {
    if (is_leaf(tree, j)) {
      w * leaf_value(tree, j)
    } else {
      aj <- lesser(tree, j)
      bj <- greater(tree, j)
      cj <- missing(tree, j)
      stopifnot(length(aj) == 1)
      stopifnot(length(bj) == 1)
      stopifnot(length(cj) == 1)
      if (feature(tree, j) %in% S) {
        if (is.na(x[[feature(tree, j)]])) {
          if (!is.na(cj)) {
            G(cj, w)
          } else {
            stop("model does not work with NAs!")
          }
        } else if (x[[feature(tree, j)]] <= threshold(tree, j)) {
          G(aj, w)
        } else {
          G(bj, w)
        }
      } else {
        if (is.na(cj) | cj == aj | cj == bj) {
          G(aj, w * cover(tree, aj) / cover(tree, j)) + G(bj, w * cover(tree, bj) / cover(tree, j))
        } else {
          G(aj, w * cover(tree, aj) / cover(tree, j)) + G(bj, w * cover(tree, bj) / cover(tree, j)) + G(cj, w * cover(tree, cj) / cover(tree, j))
        }
      }
    }
  }
  G(root, 1)
}

# Function coppied form rje package ver 1.10.16
powerset <- function (x, m, rev = FALSE) {
  if (base::missing(m)) m = length(x)
  if (m == 0) return(list(x[c()]))

  out = list(x[c()])
  if (length(x) == 1)
    return(c(out, list(x)))
  for (i in seq_along(x)) {
    if (rev)
      out = c(lapply(out[lengths(out) < m], function(y) c(y, x[i])), out)
    else out = c(out, lapply(out[lengths(out) < m], function(y) c(y, x[i])))
  }
  out
}

# exponential calculation of SHAP Values
shap_exponential <- function(unified_model, x) {
  model <- unified_model$model

  shaps <- data.frame()
  for (row in 1:nrow(x)) {
    m <- ncol(x)
    shaps.row <- rep(0, times = m)
    for (t in tree_ids(model)) {
      root <- extract_tree_root(model, t)
      for (var_id in 1:m) {
        oth_vars <- colnames(x)[-var_id]
        var <- colnames(x)[var_id]
        sets <- powerset(oth_vars)
        for (S in sets) {
          f_without <- expvalue(model, root, x[row, ], S)
          f_with <- expvalue(model, root, x[row, ], c(S, var))
          size <- length(S)
          weight <- factorial(size) * factorial(m - size - 1) / factorial(m)
          shaps.row[var_id] <- shaps.row[var_id] + (f_with - f_without) * weight
        }
      }
    }
    shaps <- rbind(shaps, as.data.frame(matrix(shaps.row, nrow = 1)))
  }

  colnames(shaps) <- colnames(x)
  shaps
}

# exponential calculation of SHAP Interaction Values
shap_interactions_exponential <- function(unified_model, x) {
  model <- unified_model$model

  shaps <- as.matrix(shap_exponential(unified_model, x))
  interactions_array <- array(0,
                              dim = c(ncol(x), ncol(x), nrow(x)),
                              dimnames = list(colnames(x), colnames(x), c()))
  for (row in 1:nrow(x)) {
    m <- ncol(x)
    for (t in tree_ids(model)) {
      root <- extract_tree_root(model, t)
      for (var1_id in 1:m) {
        for (var2_id in 1:m) {
          if (var1_id < var2_id) {
            oth_vars <- colnames(x)[-c(var1_id, var2_id)]
            var1 <- colnames(x)[var1_id]
            var2 <- colnames(x)[var2_id]
            sets <- powerset(oth_vars)
            for (S in sets) {
              f_without_both <- expvalue(model, root, x[row, ], S)
              f_without_1 <- expvalue(model, root, x[row, ], c(S, var2))
              f_without_2 <- expvalue(model, root, x[row, ], c(S, var1))
              f_with <- expvalue(model, root, x[row, ], c(S, var1, var2))
              size <- length(S)
              weight <- factorial(size) * factorial(m - size - 2) / (2 * factorial(m - 1))
              sum <- weight * (f_with + f_without_both - f_without_1 - f_without_2)

              interactions_array[var1_id, var2_id, row] <- interactions_array[var1_id, var2_id, row] + sum
              interactions_array[var2_id, var1_id, row] <- interactions_array[var2_id, var1_id, row] + sum
            }
          }
        }
      }
    }

    # filling the diagonal
    row_sums <- apply(interactions_array[, , row], 2, sum)
    diag(interactions_array[, , row]) <- shaps[row, ] - row_sums
  }

  interactions_array
}



treeshap_correctness_test <- function(max_depth, nrounds, nobservations,
                                      model = "xgboost", test_data = data, test_target = target) {
  model <- test_model(max_depth, nrounds, model, test_data, test_target)
  set.seed(21)
  rows <- sample(nrow(test_data), nobservations)
  shaps_exp <- shap_exponential(model, test_data[rows, ])
  treeshap_res <- treeshap(model, test_data[rows, ], verbose = FALSE)
  shaps_treeshap <- treeshap_res$shaps

  precision <- 4
  is.treeshap(treeshap_res) & all(round(shaps_exp, precision) == round(shaps_treeshap, precision))
}

interactions_correctness_test <- function(max_depth, nrounds, nobservations,
                                          model = "xgboost", test_data = data, test_target = target) {
  model <- test_model(max_depth, nrounds, model, test_data, test_target)
  set.seed(21)
  rows <- sample(nrow(test_data), nobservations)
  interactions_exp <- shap_interactions_exponential(model, test_data[rows, ])
  treeshap_res <- treeshap(model, test_data[rows, ], interactions = TRUE, verbose = FALSE)
  interactions_treeshap <- treeshap_res$interactions

  precision_relative <- 1e-08
  precision_absolute <- 1e-08
  relative_error <- abs((interactions_exp - interactions_treeshap) / interactions_exp) < precision_relative
  absolute_error <- abs(interactions_exp - interactions_treeshap) < precision_absolute
  error <- relative_error | absolute_error
  is.treeshap(treeshap_res) & all(error)
}

shaps_sum_test <- function(max_depth, nrounds, nobservations,
                           model_type = "xgboost", test_data = data, test_target = target, precision = 1e-12) {
  model <- test_model(max_depth, nrounds, model_type, test_data, test_target)
  set.seed(21)
  rows <- sample(nrow(test_data), nobservations)

  ntrees <- sum(model$model$Node == 0)
  leaves <- model$model[is.na(model$model$Feature), ]
  intercept <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees
  prediction <- predict(model, test_data[rows, ])
  prediction_deviation <- prediction - intercept

  treeshap_res <- treeshap(model, test_data[rows, ], interactions = TRUE, verbose = FALSE)

  basic_shaps_sum <- apply(treeshap_res$shaps, 1, sum)
  expect_true(all(abs((prediction_deviation - basic_shaps_sum) / prediction_deviation) < precision))

  interactions_sum <- apply(treeshap_res$interactions, 3, sum)
  expect_true(all(abs((prediction_deviation - interactions_sum) / prediction_deviation) < precision))
}

test_that("treeshap function checks", {
  library(lightgbm)
  param_lgbm <- list(objective = "regression", max_depth = 2,  force_row_wise = TRUE)
  data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
               c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
               'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]

  data_df <- as.matrix(na.omit(cbind(data_fifa, fifa20$target)))
  sparse_data <- data_df[,-ncol(data_df)]
  x <- lightgbm::lgb.Dataset(sparse_data, label = data_df[,ncol(data_df)])
  lgb_data <- lightgbm::lgb.Dataset.construct(x)
  lgb_model <- lightgbm::lightgbm(data = lgb_data, params = param_lgbm, verbose = -1,
                                  save_name = paste0(tempfile(), '.model'))
  unified_model <- lightgbm.unify(lgb_model, sparse_data)
  expect_error(treeshap(unified_model, sparse_data[1:2,], verbose = FALSE))
})


test_that('treeshap correctness test 1 (xgboost, max_depth = 3, nrounds = 1, nobservations = 25)', {
  expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 1, nobservations = 25, model = "xgboost"))
})

test_that('treeshap correctness test 2 (xgboost, max_depth = 12, nrounds = 3, nobservations = 5)', {
  expect_true(treeshap_correctness_test(max_depth = 12, nrounds = 3, nobservations = 5, model = "xgboost"))
})

test_that('treeshap correctness test 3 (xgboost, max_depth = 7, nrounds = 7, nobservations = 3, with NAs)', {
  expect_true(treeshap_correctness_test(max_depth = 7, nrounds = 7, nobservations = 3, model = "xgboost", test_data = data_na))
})

test_that('treeshap correctness test 4 (ranger, max_depth = 5, nrounds = 7, nobservations = 5)', {
  expect_true(treeshap_correctness_test(max_depth = 5, nrounds = 7, nobservations = 5, model = "ranger"))
})

test_that('treeshap correctness test 5 (randomForest, max_depth = 3, nrounds = 7, nobservations = 5)', {
  expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 7, nobservations = 5, model = "randomForest"))
})

test_that('treeshap correctness test 6 (gbm, max_depth = 3, nrounds = 7, nobservations = 5, with NAs)', {
  expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 7, nobservations = 5, model = "gbm", test_data = data_na))
})

test_that('treeshap correctness test 7 (lightgbm, max_depth = 3, nrounds = 7, nobservations = 5, with NAs)', {
  expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 7, nobservations = 5, model = "lightgbm", test_data = data_na))
})

# test_that('treeshap correctness test 8 (catboost, max_depth = 3, nrounds = 7, nobservations = 5, with NAs)', {
#   expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 7, nobservations = 5, model = "catboost", test_data = data_na))
# }) # TODO for some reason exponential calculation returns NA for higher max_depth or nrounds than like(2, 4)



test_that('interactions correctness test 1 (xgboost, max_depth = 3, nrounds = 1, nobservations = 25)', {
  expect_true(interactions_correctness_test(max_depth = 3, nrounds = 1, nobservations = 25, model = "xgboost"))
})

test_that('interactions correctness test 2 (xgboost, max_depth = 12, nrounds = 3, nobservations = 5)', {
  expect_true(interactions_correctness_test(max_depth = 12, nrounds = 3, nobservations = 5, model = "xgboost"))
})

test_that('interactions correctness test 3 (xgboost, max_depth = 7, nrounds = 4, nobservations = 2, with NAs)', {
  expect_true(interactions_correctness_test(max_depth = 7, nrounds = 4, nobservations = 2, model = "xgboost", test_data = data_na))
})

test_that('interactions correctness test 4 (ranger, max_depth = 6, nrounds = 4, nobservations = 2)', {
  expect_true(interactions_correctness_test(max_depth = 6, nrounds = 4, nobservations = 2, model = "ranger"))
})

test_that('interactions correctness test 5 (randomForest, max_depth = 4, nrounds = 4, nobservations = 2)', {
  expect_true(interactions_correctness_test(max_depth = 4, nrounds = 4, nobservations = 2, model = "randomForest"))
})

test_that('interactions correctness test 6 (gbm, max_depth = 4, nrounds = 4, nobservations = 2, with NAs)', {
  expect_true(interactions_correctness_test(max_depth = 4, nrounds = 4, nobservations = 2, model = "gbm", test_data = data_na))
})

test_that('interactions correctness test 7 (lightgbm, max_depth = 4, nrounds = 4, nobservations = 2, with NAs)', {
  expect_true(interactions_correctness_test(max_depth = 4, nrounds = 4, nobservations = 2, model = "lightgbm", test_data = data_na))
})

# test_that('interactions correctness test 8 (catboost, max_depth = 4, nrounds = 5, nobservations = 2, with NAs)', {
#   expect_true(interactions_correctness_test(max_depth = 4, nrounds = 5, nobservations = 2, model = "catboost", test_data = data_na))
# }) # !!! TODO for some reason exponential calculation returns NA for higher max_depth or nrounds than like (2, 2)

test_that('xgboost: shaps sum up to prediction deviation (max_depth = 6, nrounds = 100, nobservations = 40, with NAs)', {
  shaps_sum_test(model_type = "xgboost", max_depth = 6, nrounds = 100, nobservations = 40, test_data = data_na)
})

test_that('ranger: shaps sum up to prediction deviation (max_depth = 6, nrounds = 100, nobservations = 40)', {
  shaps_sum_test(model_type = "ranger", max_depth = 6, nrounds = 100, nobservations = 40)
})

test_that('randomForest: shaps sum up to prediction deviation (max_depth = 6, nrounds = 100, nobservations = 40)', {
  shaps_sum_test(model_type = "randomForest", max_depth = 6, nrounds = 100, nobservations = 40)
})

test_that('gbm: shaps sum up to prediction deviation (max_depth = 6, nrounds = 40, nobservations = 40, with NAs)', {
  shaps_sum_test(model_type = "gbm", max_depth = 6, nrounds = 40, nobservations = 40, test_data = data_na)
})

test_that('lightgbm: shaps sum up to prediction deviation (max_depth = 6, nrounds = 100, nobservations = 40, with NAs)', {
  shaps_sum_test(model_type = "lightgbm", max_depth = 4, nrounds = 100, nobservations = 40, test_data = data_na)
})

test_that('catboost: shaps sum up to prediction deviation (max_depth = 6, nrounds = 100, nobservations = 40, with NAs)', {
  shaps_sum_test(model_type = "catboost", max_depth = 6, nrounds = 100, nobservations = 40, test_data = data_na)
})
