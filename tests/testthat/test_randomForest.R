library(treeshap)

library(randomForest)
data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
                           c('value_eur', 'gk_diving', 'gk_handling',
                             'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
x <- na.omit(cbind(data_fifa, target = fifa20$target))

rf_with_cat_model <- randomForest::randomForest(target~., data = x, maxnodes = 10, ntree = 10)

x <- x[colnames(x) != 'work_rate']


rf_num_model <- randomForest::randomForest(target~., data = x, maxnodes = 10, ntree = 10)


test_that('randomForest.unify returns an object with correct attributes', {
  unified_model <- randomForest.unify(rf_num_model, x)

  expect_equal(attr(unified_model, "missing_support"), FALSE)
  expect_equal(attr(unified_model, "model"), "randomForest")
})

test_that('the randomForest.unify function returns data frame with columns of appropriate column', {
  unifier <- randomForest.unify(rf_num_model, x)$model
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
  unifier <- randomForest.unify(rf_num_model, x)
  expect_error(treeshap(unifier, x[1:3,], verbose = FALSE), NA)
})


