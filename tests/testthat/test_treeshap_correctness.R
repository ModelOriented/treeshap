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

powerset <- rje::powerSet

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


correctness_test <- function(max_depth, nrounds, nobservations) {
  model <- test_model(max_depth, nrounds)
  set.seed(21)
  rows <- sample(nrow(fifa20$data), nobservations)
  shaps_exp <- shap_exponential(model, data[rows, ])
  shaps_treeshap <- treeshap(model, data[rows, ])
  precision <- 4
  dplyr::all_equal(round(shaps_exp, precision), round(shaps_treeshap, precision))
}

test_that('correctness test 1 (max_depth = 3, nrounds = 1, nobservations = 25)', {
  expect_true(correctness_test(max_depth = 3, nrounds = 1, nobservations = 25))
})

test_that('correctness test 2 (max_depth = 12, nrounds = 3, nobservations = 5)', {
  expect_true(correctness_test(max_depth = 12, nrounds = 3, nobservations = 5))
})

test_that('correctness test 3 (max_depth = 7, nrounds = 10, nobservations = 3)', {
  expect_true(correctness_test(max_depth = 7, nrounds = 10, nobservations = 3))
})
