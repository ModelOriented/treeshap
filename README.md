
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
supports models produced with `XGBoost`, `LightGBM`, `GBM` and
`Catboost`.

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
#>        Tree Node   Feature Split Yes No Missing Quality/Score Cover
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
treeshap_values <- treeshap(unified,  fifa20$data[700:800, colnames(fifa20$data) != 'work_rate'])
head(treeshap_values[,1:6])
#>          age height_cm weight_kg overall potential international_reputation
#> 1   297154.4  5769.186 12136.316 8739757  212428.8               -50855.738
#> 2 -2550066.6 16011.136  3134.526 6525123  244814.2                22784.430
#> 3   300830.3 -9023.299 15374.550 8585145  479118.8                 2374.351
#> 4  -132645.2 12380.183 33731.893 8321266  357679.5                49019.904
#> 5   689402.9 -3369.050 16433.595 8933670  427577.5                12147.246
#> 6 -1042288.0  5760.739  8428.627 6579288  289577.8                66873.547
```

## How fast does it work?

Complexity of TreeSHAP is `O(TLD^2)`, where `T` is number of trees, `L`
is number of leaves in a tree and `D` is depth of a tree.

Our implementation works in speed comparable to original Lundberg’s
Python package `shap` implementation using C and Python.

In the following example our TreeSHAP implementation explains 300
observations on a model consisting of 200 trees of max depth = 6 in less
than 2 seconds.

``` r
microbenchmark::microbenchmark(
  treeshap = treeshap(unified,  fifa20$data[1:300, colnames(fifa20$data) != 'work_rate']),
  times = 5
)
#> Unit: seconds
#>      expr      min      lq     mean   median       uq      max neval
#>  treeshap 1.951773 1.98665 2.016904 1.997519 2.012991 2.135587     5
```

## How to use the unifying functions?

Even though the data frames produced by the functions from `.unify()`
family (`xgboost.unify()`, `lightgbm.unify()`, `gbm.unify()`,
`catboost.unify()`) are identical when it comes to the structure, due to
different possibilities of saving and representing the trees among the
packages, the usage of functions is slightly different. As an argument,
first three listed functions take an object of appropriate model. The
latter one, `catboost.unify()` requires a transformed dataset used for
training the model - an object of class `catboost.Pool`. Here is a short
example representing usage of two functions:

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
#>    Tree Node   Feature Split Yes No Missing Quality/Score Cover
#> 1:    0    0   overall  65.5   2  3       7     5367.6866  9139
#> 2:    0    1      <NA>    NA  NA NA      NA   -37500.0000  4231
#> 3:    0    2 potential  68.5   4  5       6      995.1723  4908
#> 4:    0    3      <NA>    NA  NA NA      NA    -5000.0000   766
#> 5:    0    4      <NA>    NA  NA NA      NA   170000.0000  4142
#> 6:    0    5      <NA>    NA  NA NA      NA   142687.4491  4908
```

#### 2\. Catboost

For representing correct names of features that are regarding during
splitting observations into sets, `catboost.unify()` requires passing
two arguments. Some values (Quality/Score) are unavailable for internal
nodes in the data frame created on catboost model:

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
