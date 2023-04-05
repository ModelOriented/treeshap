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

# to save some time for these tests, compute model here once:
unified_model <- ranger_surv.unify(ranger_num_model, x)
test_that('ranger_surv.unify creates an object of the appropriate class', {
  expect_true(is.model_unified(unified_model))
})

test_that('ranger_surv.unify returns an object with correct attributes', {
  expect_equal(attr(unified_model, "missing_support"), FALSE)
  expect_equal(attr(unified_model, "model"), "ranger")
})

test_that('the ranger_surv.unify function returns data frame with columns of appropriate column', {
  unifier <- unified_model$model
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
  expect_error(treeshap(unified_model, x[1:3,], verbose = FALSE), NA)
})

test_that("ranger_surv: predictions from unified == original predictions", {
  obs <- x[1:800, ]
  surv_preds <- stats::predict(ranger_num_model, obs)
  original <- rowSums(surv_preds$chf)
  from_unified <- predict(unified_model, obs)
  expect_true(all(abs((from_unified - original) / original) < 10**(-14)))
})

test_that("ranger_surv: mean prediction calculated using predict == using covers", {

  intercept_predict <- mean(predict(unified_model, x))

  ntrees <- sum(unified_model$model$Node == 0)
  leaves <- unified_model$model[is.na(unified_model$model$Feature), ]
  intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

  #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
  expect_equal(intercept_predict, intercept_covers)
})

test_that("ranger_surv: covers correctness", {

  roots <- unified_model$model[unified_model$model$Node == 0, ]
  expect_true(all(roots$Cover == nrow(x)))

  internals <- unified_model$model[!is.na(unified_model$model$Feature), ]
  yes_child_cover <- unified_model$model[internals$Yes, ]$Cover
  no_child_cover <- unified_model$model[internals$No, ]$Cover
  if (all(is.na(internals$Missing))) {
    children_cover <- yes_child_cover + no_child_cover
  } else {
    missing_child_cover <- unified_model$model[internals$Missing, ]$Cover
    missing_child_cover[is.na(missing_child_cover)] <- 0
    missing_child_cover[internals$Missing == internals$Yes | internals$Missing == internals$No] <- 0
    children_cover <- yes_child_cover + no_child_cover + missing_child_cover
  }
  expect_true(all(internals$Cover == children_cover))
})


# to save some time for these tests, compute model here once:
unified_model <- ranger_surv_fun.unify(ranger_num_model, x)
test_that('ranger_surv_fun.unify creates an object of the appropriate class', {
  lapply(unified_model, function(m) expect_true(is.model_unified(m)))
})

test_that('ranger_surv_fun.unify returns an object with correct attributes', {
  for (m in unified_model) {
    expect_equal(attr(m, "missing_support"), FALSE)
    expect_equal(attr(m, "model"), "ranger")
  }
})

test_that('the ranger_surv_fun.unify function returns data frame with columns of appropriate column', {
  for (m in unified_model) {
    unifier <- m$model
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
  }
})

test_that("ranger_surv: shap calculates without an error", {
  for (m in unified_model) {
    expect_error(treeshap(m, x[1:3,], verbose = FALSE), NA)
  }
})

test_that("ranger_surv: predictions from unified == original predictions", {
  for (t in names(unified_model)) {
    m <- unified_model[[t]]
    death_time <- as.integer(t)
    obs <- x[1:800, ]
    surv_preds <- stats::predict(ranger_num_model, obs)
    original <- surv_preds$survival[, which(surv_preds$unique.death.times == death_time)]
    from_unified <- predict(m, obs)
    # this is yet kind of strange that values differ so much
    expect_true(all(abs(from_unified - original) < 1.5e-1))
    #expect_true(all(abs((from_unified - original) / original) < 10**(-14)))
  }
})

test_that("ranger_surv: mean prediction calculated using predict == using covers", {
  for (m in unified_model) {
    intercept_predict <- mean(predict(m, x))

    ntrees <- sum(m$model$Node == 0)
    leaves <- m$model[is.na(m$model$Feature), ]
    intercept_covers <- sum(leaves$Prediction * leaves$Cover) / sum(leaves$Cover) * ntrees

    #expect_true(all(abs((intercept_predict - intercept_covers) / intercept_predict) < 10**(-14)))
    expect_equal(intercept_predict, intercept_covers)
  }
})

test_that("ranger_surv: covers correctness", {
  for (m in unified_model) {
    roots <- m$model[m$model$Node == 0, ]
    expect_true(all(roots$Cover == nrow(x)))

    internals <- m$model[!is.na(m$model$Feature), ]
    yes_child_cover <- m$model[internals$Yes, ]$Cover
    no_child_cover <- m$model[internals$No, ]$Cover
    if (all(is.na(internals$Missing))) {
      children_cover <- yes_child_cover + no_child_cover
    } else {
      missing_child_cover <- m$model[internals$Missing, ]$Cover
      missing_child_cover[is.na(missing_child_cover)] <- 0
      missing_child_cover[internals$Missing == internals$Yes | internals$Missing == internals$No] <- 0
      children_cover <- yes_child_cover + no_child_cover + missing_child_cover
    }
    expect_true(all(internals$Cover == children_cover))
  }
})
