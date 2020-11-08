## Small tests comparing TreeSHAP results to brutal implementation results
## Extensive testing of correctness is not possible due to complexity of brutal implementation

library(treeshap)

data <- fifa20$data[colnames(fifa20$data) != 'work_rate']

# limiting columns for faster exponential calculation
data <- data[, 3:6]

test_model <- function(max_depth, nrounds) {
  target <- fifa20$target
  param <- list(objective = "reg:squarederror", max_depth = max_depth)
  xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = nrounds, verbose = FALSE)
  xgboost.unify(xgb_model)
}


## Implementation of exponential complexity SHAP calculation

# functions wrapping tree structure
is_leaf <- function(model, j) (is.na(model$Feature[j]))
leaf_value <- function(model, j) {
  stopifnot(is_leaf(model, j))
  model[[j, "Quality/Score"]]
}
feature <- function(model, j) (model$Feature[j])
lesser <- function(model, j) (model$Yes[j])
greater <- function(model, j) (model$No[j])
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
      stopifnot(length(aj) == 1)
      stopifnot(length(bj) == 1)
      if (feature(tree, j) %in% S) {
        if (x[[feature(tree, j)]] <= threshold(tree, j)) {
          G(aj, w)
        } else {
          G(bj, w)
        }
      } else {
        G(aj, w * cover(tree, aj) / cover(tree, j)) + G(bj, w * cover(tree, bj) / cover(tree, j))
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
shap_exponential <- function(model, x) {
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
shap_interactions_exponential <- function(model, x) {
  shaps <- as.matrix(shap_exponential(model, x))
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



treeshap_correctness_test <- function(max_depth, nrounds, nobservations) {
  model <- test_model(max_depth, nrounds)
  set.seed(21)
  rows <- sample(nrow(fifa20$data), nobservations)
  shaps_exp <- shap_exponential(model, data[rows, ])
  shaps_treeshap <- treeshap(model, data[rows, ])
  precision <- 4
  dplyr::all_equal(round(shaps_exp, precision), round(shaps_treeshap, precision))
}

interactions_correctness_test <- function(max_depth, nrounds, nobservations) {
  model <- test_model(max_depth, nrounds)
  set.seed(21)
  rows <- sample(nrow(fifa20$data), nobservations)
  interactions_exp <- shap_interactions_exponential(model, data[rows, ])
  interactions_treeshap <- treeshap(model, data[rows, ], interactions = TRUE)

  precision_relative <- 1e-08
  precision_absolute <- 1e-08
  relative_error <- abs((interactions_exp - interactions_treeshap) / interactions_exp) < precision_relative
  absolute_error <- abs(interactions_exp - interactions_treeshap) < precision_absolute
  error <- relative_error | absolute_error
  all(error)
}

test_that("treeshap function checks", {
  library(lightgbm)
  library(Matrix)
  param_lgbm <- list(objective = "regression", max_depth = 2,  force_row_wise = TRUE)
  data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
               c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
               'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
  data_df <- as.matrix(na.omit(data.table::as.data.table(cbind(data_fifa, fifa20$target))))
  sparse_data <- as(data_df[,-ncol(data_df)], 'sparseMatrix')
  x <- lightgbm::lgb.Dataset(sparse_data, label = as(data_df[,ncol(data_df)], 'sparseMatrix'))
  lgb_data <- lightgbm::lgb.Dataset.construct(x)
  lgb_model <- lightgbm::lightgbm(data = lgb_data, params = param_lgbm, save_name = "", verbose = 0)
  unified_model <- lightgbm.unify(lgb_model)
  expect_error(treeshap(unified_model, sparse_data[1:2,]))
})


test_that('treeshap correctness test 1 (max_depth = 3, nrounds = 1, nobservations = 25)', {
  expect_true(treeshap_correctness_test(max_depth = 3, nrounds = 1, nobservations = 25))
})

test_that('treeshap correctness test 2 (max_depth = 12, nrounds = 3, nobservations = 5)', {
  expect_true(treeshap_correctness_test(max_depth = 12, nrounds = 3, nobservations = 5))
})

test_that('treeshap correctness test 3 (max_depth = 7, nrounds = 10, nobservations = 3)', {
  expect_true(treeshap_correctness_test(max_depth = 7, nrounds = 10, nobservations = 3))
})


test_that('interactions correctness test 1 (max_depth = 3, nrounds = 1, nobservations = 25)', {
  expect_true(interactions_correctness_test(max_depth = 3, nrounds = 1, nobservations = 25))
})

test_that('interactions correctness test 2 (max_depth = 12, nrounds = 3, nobservations = 5)', {
  expect_true(interactions_correctness_test(max_depth = 12, nrounds = 3, nobservations = 5))
})

test_that('interactions correctness test 3 (max_depth = 7, nrounds = 10, nobservations = 3)', {
  expect_true(interactions_correctness_test(max_depth = 7, nrounds = 10, nobservations = 3))
})
