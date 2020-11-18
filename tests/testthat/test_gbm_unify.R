library(treeshap)

x <- fifa20$data
x['value_eur'] <- fifa20$target

gbm_with_cat_model <- gbm::gbm(
  formula = value_eur ~ .,
  data = x,
  distribution = "laplace",
  n.trees = 20,
  cv.folds = 2,
  interaction.depth = 2,
  n.cores = 1
)

x <- x[colnames(fifa20$data) != 'work_rate']

gbm_num_model <- gbm::gbm(
  formula = value_eur ~ .,
  data = x,
  n.trees = 100,
  distribution = 'gaussian',
  n.cores = 1
)


test_that('gbm.unify returns an object with correct attributes', {
  unified_model <- gbm.unify(gbm_num_model, x)

  expect_equal(attr(unified_model, "missing_support"), TRUE)
  expect_equal(attr(unified_model, "model"), "gbm")
})

test_that('the gbm.unify function does not support models with categorical features', {
  expect_error(gbm.unify(gbm_with_cat_model), "Models built on data with categorical features are not supported - please encode them before training.")
})

test_that('the gbm.unify function returns data frame with columns of appropriate column', {
  unifier <- gbm.unify(gbm_num_model, x)$model
  expect_true(is.integer(unifier$Tree))
  expect_true(is.integer(unifier$Node))
  expect_true(is.character(unifier$Feature))
  expect_true(is.numeric(unifier$Split))
  expect_true(is.integer(unifier$Yes))
  expect_true(is.integer(unifier$No))
  expect_true(is.integer(unifier$Missing))
  expect_true(is.numeric(unifier$Prediction))
  expect_true(is.numeric(unifier$Cover))
})

test_that("shap calculates without an error", {
  unifier <- gbm.unify(gbm_num_model, x)
  expect_error(treeshap(unifier, x[1:3,], verbose = FALSE), NA)
})

test_that("gbm: predictions from unified == original predictions", {
  unifier <- gbm.unify(gbm_num_model, x)
  obs <- x[1:16000, ]
  original <- stats::predict(gbm_num_model, obs, n.trees = 100)
  from_unified <- predict(unifier, obs)
  # expect_equal(from_unified, original) #there are small differences
  expect_true(all(abs((from_unified - original) / original) < 10**(-8)))
})
