library(treeshap)

data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
                           c('value_eur', 'gk_diving', 'gk_handling',
                             'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
x <- na.omit(cbind(data_fifa, target = fifa20$target))

ranger_with_cat_model <- ranger::ranger(target ~ ., data = x, max.depth = 10, num.trees = 10)

x <- x[colnames(x) != 'work_rate']


ranger_num_model <- ranger::ranger(target ~ ., data = x, max.depth = 10, num.trees = 10)


test_that('the ranger.unify function returns data frame with columns of appropriate column', {
  unifier <- ranger.unify(ranger_num_model, x)
  expect_true(is.integer(unifier$Tree))
  expect_true(is.integer(unifier$Node))
  expect_true(is.character(unifier$Feature))
  expect_true(is.numeric(unifier$Split))
  expect_true(is.integer(unifier$Yes))
  expect_true(is.integer(unifier$No))
  expect_true(all(is.na(unifier$Missing)))
  expect_true(is.numeric(unifier[['Quality/Score']]))
  expect_true(is.numeric(unifier$Cover))
})

test_that("shap calculates without an error", {
  unifier <- ranger.unify(ranger_num_model, x)
  expect_error(treeshap(unifier, x[1:3,], verbose = FALSE), NA)
})


