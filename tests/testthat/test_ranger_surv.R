library(treeshap)

data_colon <- data.table::data.table(survival::colon)
data_colon <- na.omit(data_colon[get("etype") == 2, ])
surv_cols <- c("status", "time", "rx")

feature_cols <- colnames(data_colon)[3:(ncol(data_colon) - 1)]

x <- model.matrix(
  ~ -1 + .,
  data_colon[, .SD, .SDcols = setdiff(feature_cols, surv_cols[1:2])]
)
y <- survival::Surv(
  event = (data_colon[, get("status")] |>
             as.character() |>
             as.integer()),
  time = data_colon[, get("time")],
  type = "right"
)

ranger_num_model <- ranger::ranger(
  x = x,
  y = y,
  data = data_colon,
  max.depth = 10,
  num.trees = 10
)


test_that('ranger_surv.unify creates an object of the appropriate class', {
  expect_true(is.model_unified(ranger_surv.unify(ranger_num_model, x)))
})

test_that('ranger_surv.unify returns an object with correct attributes', {
  unified_model <- ranger_surv.unify(ranger_num_model, x)

  expect_equal(attr(unified_model, "missing_support"), FALSE)
  expect_equal(attr(unified_model, "model"), "ranger")
})

test_that('the ranger_surv.unify function returns data frame with columns of appropriate column', {
  unifier <- ranger_surv.unify(ranger_num_model, x)$model
  expect_true(is.integer(unifier$Tree))
  expect_true(is.integer(unifier$Node))
  expect_true(is.character(unifier$Feature))
  expect_true(is.factor(unifier$Decision.type))
  expect_true(is.numeric(unifier$Split))
  expect_true(is.integer(unifier$Yes))
  expect_true(is.integer(unifier$No))
  expect_true(all(is.na(unifier$Missing)))
  expect_true(is.numeric(unifier$Prediction))
  expect_true(is.numeric(unifier$Cover))
})

test_that("ranger_surv: shap calculates without an error", {
  unifier <- ranger_surv.unify(ranger_num_model, x)
  expect_error(treeshap(unifier, x[1:3,], verbose = FALSE), NA)
})

test_that("ranger_surv: predictions from unified == original predictions", {
  unifier <- ranger_surv.unify(ranger_num_model, x)
  obs <- x[1:800, ]
  surv_preds <- stats::predict(ranger_num_model, obs)
  original <- rowSums(surv_preds$chf)
  from_unified <- predict(unifier, obs)
  expect_true(all(abs((from_unified - original) / original) < 10**(-14)))
})

test_that("ranger_surv: mean prediction calculated using predict == using covers", {
  unifier <- ranger_surv.unify(ranger_num_model, x)

  intercept_predict <- mean(predict(unifier, x))

  ntrees <- sum(unifier$model$Node == 0)
  leaves <- unifier$model[is.na(unifier$model$Feature), ]
  intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

  #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
  expect_equal(intercept_predict, intercept_covers)
})

test_that("ranger_surv: covers correctness", {
  unifier <- ranger_surv.unify(ranger_num_model, x)

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
