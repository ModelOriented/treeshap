#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
NumericVector predict_cpp(DataFrame x, DataFrame is_na, IntegerVector roots, IntegerVector yes, IntegerVector no,
                          IntegerVector missing, LogicalVector is_leaf, IntegerVector feature, NumericVector split,
                          IntegerVector decision_type, NumericVector value) {
  NumericVector prediction(x.ncol());
  for (int i = 0; i < x.ncol(); ++i) {
    NumericVector observation = x[i];
    LogicalVector observation_is_na = is_na[i];

    for (int node: roots) {
      while (!is_leaf[node]) {
        if (observation_is_na[feature[node]]) {
          node = missing[node];
        } else if (((decision_type[node] == 1) && (observation[feature[node]] <= split[node]))
                     || ((decision_type[node] == 2) && (observation[feature[node]] < split[node]))) {
          node = yes[node];
        } else {
          node = no[node];
        }
      }
      prediction[i] += value[node];
    }
  }

  return prediction;
}

