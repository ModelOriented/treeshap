library(treeshap)

data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
                           c('value_eur', 'gk_diving', 'gk_handling',
                             'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
x <- na.omit(cbind(data_fifa, target = fifa20$target))

ranger_with_cat_model <- ranger::ranger(target ~ ., data = x, max.depth = 10, num.trees = 10)

x <- x[colnames(x) != 'work_rate']


ranger_num_model <- ranger::ranger(target ~ ., data = x, max.depth = 10, num.trees = 10)


test_that('ranger.unify returns an object with correct attributes', {
  unified_model <- ranger.unify(ranger_num_model, x)

  expect_equal(attr(unified_model, "missing_support"), FALSE)
  expect_equal(attr(unified_model, "model"), "ranger")
})

test_that('the ranger.unify function returns data frame with columns of appropriate column', {
  unifier <- ranger.unify(ranger_num_model, x)$model
  expect_true(is.integer(unifier$Tree))
  expect_true(is.integer(unifier$Node))
  expect_true(is.character(unifier$Feature))
  expect_true(is.numeric(unifier$Split))
  expect_true(is.integer(unifier$Yes))
  expect_true(is.integer(unifier$No))
  expect_true(all(is.na(unifier$Missing)))
  expect_true(is.numeric(unifier$Prediction))
  expect_true(is.numeric(unifier$Cover))
})

test_that("shap calculates without an error", {
  unifier <- ranger.unify(ranger_num_model, x)
  expect_error(treeshap(unifier, x[1:3,], verbose = FALSE), NA)
})

test_that("ranger: predictions from unified == original predictions", {
  unifier <- ranger.unify(ranger_num_model, x)
  obs <- x[1:16000, ]
  original <- ranger::predictions(stats::predict(ranger_num_model, obs))
  from_unified <- predict(unifier, obs)
  expect_true(all(abs((from_unified - original) / original) < 10**(-14)))
})


