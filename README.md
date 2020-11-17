
<!-- README.md is generated from README.Rmd. Please edit that file -->

# treeshap

<!-- badges: start -->

<!-- badges: end -->

In the era of complicated classifiers conquering their market, sometimes
even the authors of algorithms do not know the exact manner of building
a tree ensemble model. The difficulties in models’ structures are one of
the reasons why most users use them simply like black-boxes. But, how
can they know whether the prediction made by the model is reasonable?
`treeshap` is an efficient answer for this question. Due to implementing
an optimised alghoritm for tree ensemble models, it calculates the SHAP
values in polynomial (instead of exponential) time. This metric is the
only possible way to measure the influence of every feature regardless
of the permutation of features. Moreover, `treeshap` package shares a
bunch of functions to unify the structure of a model. Currently it
supports models produced with `XGBoost`, `LightGBM`, `GBM`, `Catboost`,
`ranger` and `randomForest`.

## Installation

You can install the released version of treeshap using package
`devtools` with:

``` r
devtools::install_github('ModelOriented/treeshap')
```

## Example

First of all, let’s focus on an example how to represent a `xgboost`
model as a unified data frame:

``` r
library(treeshap)
library(xgboost)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
target <- fifa20$target
param <- list(objective = "reg:squarederror", max_depth = 6)
xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200, verbose = 0)
unified <- xgboost.unify(xgb_model)
unified
#>        Tree Node   Feature Split Yes No Missing  Prediction   Cover
#>     1:    0    0   overall  81.5   2  3       2  3.060549e+17 18278
#>     2:    0    1   overall  73.5   4  5       4  1.130790e+17 17949
#>     3:    0    2   overall  84.5   6  7       6  4.021330e+16   329
#>     4:    0    3   overall  69.5   8  9       8  1.165151e+16 15628
#>     5:    0    4 potential  79.5  10 11      10  1.813107e+16  2321
#>    ---                                                             
#> 17850:  199   50      <NA>    NA  NA NA      NA -2.284977e+03     8
#> 17851:  199   51      <NA>    NA  NA NA      NA -1.966713e+03   167
#> 17852:  199   52      <NA>    NA  NA NA      NA -2.034138e+04    12
#> 17853:  199   53      <NA>    NA  NA NA      NA  1.711017e+04    15
#> 17854:  199   54      <NA>    NA  NA NA      NA  4.351900e+02    44
```

Having the data frame of unified structure, it is a piece of cake to
produce shap values of a prediction for a specific observation. The
`treeshap()` function requires passing two data frames: one representing
an ensemble model and one with the observations about which we want to
get the explanation. Obviously, the latter one should contain the same
columns as data used during building the model.

``` r
treeshap_values <- treeshap(unified,  data[700:800, ])
head(treeshap_values[, 1:6])
#>          age height_cm weight_kg overall potential international_reputation
#> 1   297154.4  5769.186 12136.316 8739757  212428.8               -50855.738
#> 2 -2550066.6 16011.136  3134.526 6525123  244814.2                22784.430
#> 3   300830.3 -9023.299 15374.550 8585145  479118.8                 2374.351
#> 4  -132645.2 12380.183 33731.893 8321266  357679.5                49019.904
#> 5   689402.9 -3369.050 16433.595 8933670  427577.5                12147.246
#> 6 -1042288.0  5760.739  8428.627 6579288  289577.8                66873.547
```

We can also compute SHAP values for interactions. As an example we will
calculate them for a model built with simpler (only 5 columns) data.

``` r
data2 <- fifa20$data[, 1:5]
xgb_model2 <- xgboost::xgboost(as.matrix(data2), params = param, label = target, nrounds = 200, verbose = 0)
unified2 <- xgboost.unify(xgb_model2)
treeshap_interactions <- treeshap(unified2,  data2[1:2, ], interactions = TRUE)
treeshap_interactions
#> , , 1
#> 
#>                   age  height_cm  weight_kg     overall  potential
#> age       -1886241.70   -3984.09  -96765.97   -47245.92  1034657.6
#> height_cm    -3984.09 -628797.41  -35476.11  1871689.75   685472.2
#> weight_kg   -96765.97  -35476.11 -983162.25  2546930.16  1559453.5
#> overall     -47245.92 1871689.75 2546930.16 55289985.16 12683135.3
#> potential  1034657.61  685472.23 1559453.46 12683135.27   868268.7
#> 
#> , , 2
#> 
#>                  age  height_cm  weight_kg    overall  potential
#> age       -2349987.9  306165.41  120483.91 -9871270.0  960198.02
#> height_cm   306165.4  -78810.31  -48271.61  -991020.7  -44632.74
#> weight_kg   120483.9  -48271.61  -21657.14  -615688.2 -380810.70
#> overall   -9871270.0 -991020.68 -615688.21 57384425.2 9603937.05
#> potential   960198.0  -44632.74 -380810.70  9603937.1 2994190.74
#> 
#> attr(,"class")
#> [1] "array"             "shap.interactions"
```

## Plotting results

The package currently provides 3 plotting functions that can be used

### Feature Contribution (Break-Down)

On this plot we can see how features contribute into the prediction for
a single observation. It is similar to the Break Down plot from
[iBreakDown](https://github.com/ModelOriented/iBreakDown) package, which
uses different method to approximate SHAP
values.

``` r
plot_contribution(treeshap_values[1, ], data[700, ], unified, min_max = c(0, 12000000))
```

<img src="man/figures/README-plot_contribution_example-1.png" width="100%" />

### Feature Importance

This plot shows us average absolute impact of features on the prediction
of the
model.

``` r
plot_feature_importance(treeshap_values, max_vars = 6)
```

<img src="man/figures/README-plot_importance_example-1.png" width="100%" />

### Feature Dependence

Using this plot we can see, how a single feature contributes into the
prediction depending on its
value.

``` r
plot_feature_dependence(treeshap_values, data[700:800, ], "height_cm")
```

<img src="man/figures/README-plot_dependence_example-1.png" width="100%" />

## How fast does it work?

Complexity of TreeSHAP is `O(TLD^2)`, where `T` is number of trees, `L`
is number of leaves in a tree and `D` is depth of a tree.

Our implementation works in speed comparable to original Lundberg’s
Python package `shap` implementation using C and Python.

In the following example our TreeSHAP implementation explains 300
observations on a model consisting of 200 trees of max depth = 6 in les
than 2 seconds.

``` r
microbenchmark::microbenchmark(
  treeshap = treeshap(unified,  data[1:300, ]), # using model and dataset from the example
  times = 5
)
#> Unit: seconds
#>      expr      min       lq    mean   median       uq      max neval
#>  treeshap 1.782249 1.784161 1.80818 1.792438 1.812077 1.869973     5
```

Complexity of SHAP interaction values computation is `O(MTLD^2)`, where
`M` is number of variables in explained dataset, `T` is number of trees,
`L` is number of leaves in a tree and `D` is depth of a tree.

SHAP Interaction values for 5 variables, model consisting of 200 trees
of max depth = 6 and 300 observations can be computed in less than 8
seconds.

``` r
microbenchmark::microbenchmark(
  treeshap = treeshap(unified2, data2[1:300, ], interactions = TRUE), # using model and dataset from the example
  times = 5
)
#> Unit: seconds
#>      expr      min       lq     mean   median      uq      max neval
#>  treeshap 7.088556 7.135377 7.139961 7.136265 7.14475 7.194855     5
```

## How to use the unifying functions?

Even though the data frames produced by the functions from `.unify()`
family (`xgboost.unify()`, `lightgbm.unify()`, `gbm.unify()`,
`catboost.unify()`, `randomForest.unify()`, `ranger.unify()`) are
identical when it comes to the structure, due to different possibilities
of saving and representing the trees among the packages, the usage of
functions is slightly different. As an argument, first three listed
functions take an object of appropriate model. The latter one,
`catboost.unify()` requires a transformed dataset used for training the
model - an object of class `catboost.Pool`. Both `randomForest.unify()`
and `ranger.unify()` also require a dataset to calculate Cover values,
usually dataset used for training the model. Here is a short example
representing usage of two functions:

#### 1\. GBM

An argument of `gbm.unify()` should be of `gbm` class - a gradient
boosting model.

``` r
library(gbm)
#> Loaded gbm 2.1.5
library(treeshap)
x <- fifa20$data[colnames(fifa20$data) != 'work_rate']
x['value_eur'] <- fifa20$target
gbm_model <- gbm::gbm(
  formula = value_eur ~ .,
  data = x,
  distribution = "laplace",
  n.trees = 200,
  cv.folds = 2,
  interaction.depth = 2
)
head(gbm.unify(gbm_model))
#>    Tree Node   Feature Split Yes No Missing    Prediction Cover
#> 1:    0    0   overall  65.5   2  3       7       5484.52  9139
#> 2:    0    1      <NA>    NA  NA NA      NA     -37500.00  4230
#> 3:    0    2 potential  68.5   4  5       6       1017.26  4909
#> 4:    0    3      <NA>    NA  NA NA      NA      -7500.00   747
#> 5:    0    4      <NA>    NA  NA NA      NA     170000.00  4162
#> 6:    0    5      <NA>    NA  NA NA      NA     142989.92  4909
```

#### 2\. Catboost

For representing correct names of features that are regarding during
splitting observations into sets, `catboost.unify()` requires passing
two arguments:

``` r
library(treeshap)
library(catboost)
data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
label <- fifa20$target
dt.pool <- catboost::catboost.load_pool(data = as.data.frame(lapply(data, as.numeric)), label = label)
cat_model <- catboost::catboost.train(
            dt.pool,
            params = list(loss_function = 'RMSE', iterations = 100, metric_period = 10,
                          logging_level = 'Silent', allow_writing_files = FALSE))
head(catboost.unify(cat_model, dt.pool))
```

## References

  - Scott M. Lundberg, Gabriel G. Erion, Su-In Lee, “Consistent
    Individualized Feature Attribution for Tree Ensembles”, University
    of Washington
