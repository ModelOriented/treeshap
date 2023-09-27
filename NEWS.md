treeshap 0.2.0
----------------------------------------------------------------
* Initial CRAN submission.
* Added implementation for random survival forests created using `ranger` ([#22](https://github.com/ModelOriented/treeshap/pull/22), [#26](https://github.com/ModelOriented/treeshap/pull/26))
* Fixed GitHub Actions, check and test issues ([#25](https://github.com/ModelOriented/treeshap/pull/25), [#29](https://github.com/ModelOriented/treeshap/pull/29)) 

treeshap 0.1.1
----------------------------------------------------------------
* Fixed `plot_contribution` when `max_vars` is larger than the number of variables ([#16](https://github.com/ModelOriented/treeshap/issues/16))

treeshap 0.1.0
----------------------------------------------------------------
* Rebuilded treeshap function so it now stores observations and whole dataset
* Rebuilded all unifiers so they require passing data.

treeshap 0.0.1
----------------------------------------------------------------
* Made package pass all checks
* Fixed infinite recursion issue in ranger  ([see commit](https://github.com/ModelOriented/treeshap/commit/eff70d8095932128151fb4c015fd61b89635aa9e))
* If there is no missing value in the model, unifiers return `NA` for `Missing` column ([see commit](https://github.com/ModelOriented/treeshap/commit/eff70d8095932128151fb4c015fd61b89635aa9e))


treeshap 0.0.0.9000
----------------------------------------------------------------
* treeshap is now public
* Implemented fast computations of tree ensemble shap values in C++
* Implemented unifiers for catboost, lightgbm, xgboost, gbm, ranger and randomForest



