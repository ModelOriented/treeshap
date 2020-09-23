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

simple_interaction_model <- read.csv("manual_tests/ultra_simple_dataset/interaction_tree.csv")
colnames(simple_interaction_model)[8] <- "Quality/Score"
simple_no_interaction_model <- read.csv("manual_tests/ultra_simple_dataset/no_interaction_tree.csv")
colnames(simple_no_interaction_model)[8] <- "Quality/Score"
simple_dataset <- read.csv("manual_tests/ultra_simple_dataset/dataset.csv")

treeshap(simple_interaction_model, simple_dataset, interactions = TRUE)
treeshap(simple_interaction_model, simple_dataset)

treeshap(simple_no_interaction_model, simple_dataset, interactions = TRUE)
treeshap(simple_no_interaction_model, simple_dataset)


shap_interactions_offdiagonal_exponential(simple_interaction_model, simple_dataset)
shap_interactions_offdiagonal_exponential(simple_no_interaction_model, simple_dataset)

# 2nd simple interaction dataset, based on titanic

simple_no_interaction_model2 <- read.csv("manual_tests/ultra_simple_dataset2/interaction_tree.csv")
colnames(simple_no_interaction_model2)[8] <- "Quality/Score"
simple_dataset2 <- read.csv("manual_tests/ultra_simple_dataset2/dataset.csv")

treeshap(simple_no_interaction_model2, simple_dataset2[1:2, ])
treeshap(simple_no_interaction_model2, simple_dataset2[2, ], interactions = TRUE)
shap_interactions_exponential(simple_no_interaction_model2, simple_dataset2[2, ])


interactions_correctness_test(max_depth = 5, nrounds = 20, nobservations = 6)


library(fastshap)
data(mtcars)
mtcars.ppr <- ppr(mpg ~ ., data = mtcars, nterms = 1)
set.seed(101)  # for reproducibility
shap <- explain(mtcars.ppr, X = subset(mtcars, select = -mpg), nsim = 10,
                pred_wrapper = predict)
shap
library(ggplot2)
autoplot(shap, X = mtcars, type = "dependence")
