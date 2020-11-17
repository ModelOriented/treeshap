library(treeshap)
library(xgboost)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
target <- fifa20$target

test_that('recalculate covers works correctly for xgboost model', {
  param <- list(objective = "reg:squarederror", max_depth = 5)
  xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 100, verbose = FALSE)
  unified <- xgboost.unify(xgb_model, as.matrix(data))
  a <- set_reference_dataset(unified, data)$Cover
  b <- unified$Cover
  expect_true(all(a == b))
})
