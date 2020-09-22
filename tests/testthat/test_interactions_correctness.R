## Small tests comparing TreeSHAP interactions results to brutal implementation results
## Extensive testing of correctness is not possible due to complexity of brutal implementation

library(treeshap)

data <- fifa20$data[colnames(fifa20$data) != 'work_rate']

## Implementation of exponential complexity SHAP interactions calculation

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

shap_interactions_offdiagonal_exponential <- function(model, x) {
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

              print(sum)

              interactions_array[var1_id, var2_id, row] <- interactions_array[var1_id, var2_id, row] + sum
              interactions_array[var2_id, var1_id, row] <- interactions_array[var2_id, var1_id, row] + sum
            }
          }
        }
      }
    }
  }

  interactions_array
}

