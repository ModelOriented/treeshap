library(treeshap)

library(xgboost)
data2 <- fifa20$data[, 1:4]
target <- fifa20$target
param2 <- list(objective = "reg:squarederror", max_depth = 4)
xgb_model2 <- xgboost::xgboost(as.matrix(data2), params = param2, label = target, nrounds = 200)
unified_model2 <- xgboost.unify(xgb_model2)

(ints <- treeshap(unified_model2, head(data2, 1), interactions = TRUE))

apply(ints, 1, sum)
treeshap(unified_model2, head(data2, 1))

Rcpp::compileAttributes()
roxygen2::roxygenise()

# simple interaction dataset and tree (https://christophm.github.io/interpretable-ml-book/interaction.html)

simple_interaction_model <- read.csv("manual_tests/ultra_simple_dataset/interaction_tree.csv")[-1]
colnames(simple_interaction_model)[8] <- "Quality/Score"
simple_no_interaction_model <- read.csv("manual_tests/ultra_simple_dataset/no_interaction_tree.csv")[-1]
colnames(simple_no_interaction_model)[8] <- "Quality/Score"
simple_dataset <- read.csv("manual_tests/ultra_simple_dataset/dataset.csv")

treeshap(simple_interaction_model, simple_dataset, interactions = TRUE)
treeshap(simple_interaction_model, simple_dataset)

treeshap(simple_no_interaction_model, simple_dataset, interactions = TRUE)
treeshap(simple_no_interaction_model, simple_dataset)

