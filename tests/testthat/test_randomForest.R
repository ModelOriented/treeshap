library(treeshap)

data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
                           c('value_eur', 'gk_diving', 'gk_handling',
                             'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
x <- na.omit(cbind(data_fifa, target = fifa20$target))

rf_with_cat_model <- randomForest::randomForest(target~., data = x, maxnodes = 10, ntree = 10)

x <- x[colnames(x) != 'work_rate']


rf_num_model <- randomForest::randomForest(target~., data = x, maxnodes = 10, ntree = 10)


test_that('randomForest.unify creates an object of the appropriate class', {
  expect_true(is.model_unified(randomForest.unify(rf_num_model, x)))
})

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
  expect_true(is.factor(unifier$Decision.type))
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

test_that("randomForest: predictions from unified == original predictions", {
  unifier <- randomForest.unify(rf_num_model, x)
  obs <- x[1:16000, ]
  original <- stats::predict(rf_num_model, obs)
  names(original) <- NULL
  from_unified <- predict(unifier, obs)
  expect_true(all(abs((from_unified - original) / original) < 10**(-14)))
})

test_that("randomForest: mean prediction calculated using predict == using covers", {
  unifier <- randomForest.unify(rf_num_model, x)

  intercept_predict <- mean(predict(unifier, x))

  ntrees <- sum(unifier$model$Node == 0)
  leaves <- unifier$model[is.na(unifier$model$Feature), ]
  intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

  #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
  expect_equal(intercept_predict, intercept_covers)
})

test_that("randomForest: covers correctness", {
  unifier <- randomForest.unify(rf_num_model, x)

  roots <- unifier$model[unifier$model$Node == 0, ]
  expect_true(all(roots$Cover == nrow(x)))

  internals <- unifier$model[!is.na(unifier$model$Feature), ]
  yes_child_cover <- unifier$model[internals$Yes, ]$Cover
  no_child_cover <- unifier$model[internals$No, ]$Cover
  if (all(is.na(internals$Missing))) {
    children_cover <- yes_child_cover + no_child_cover
  } else {
    missing_child_cover <- unifier$model[internals$Missing, ]$Cover
    missing_child_cover[is.na(missing_child_cover)] <- 0
    missing_child_cover[internals$Missing == internals$Yes | internals$Missing == internals$No] <- 0
    children_cover <- yes_child_cover + no_child_cover + missing_child_cover
  }
  expect_true(all(internals$Cover == children_cover))
})
