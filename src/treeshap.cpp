#include <Rcpp.h>

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
#include <unistd.h>
#include <Rinterface.h>
#endif

using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
void initProgressBar() {
  std::stringstream strs;
  strs <<"|0%----|------|20%---|------|40%---|------|60%---|------|80%---|------|100%\n" <<
    "=---------------------------------------------------------------------- (0%)";
  std::string temp_str = strs.str();
  char const* char_type = temp_str.c_str();
  Rprintf("\r");
  Rprintf("%s", char_type);
  Rprintf("\r");
  R_FlushConsole();
  R_CheckUserInterrupt();
}

void updateProgressBar(int steps_done, int steps_all) {
  std::stringstream strs;
  int progress_signs = int(.5 + 70 * steps_done / steps_all);
  int progress_percent = int(.5 + 100 * steps_done / steps_all);
  strs << std::string(progress_signs + 1, '=') << std::string(70 - progress_signs, '-') << " (" << progress_percent << "%)";
  std::string temp_str = strs.str();
  char const* char_type = temp_str.c_str();
  Rprintf("\r");
  Rprintf("%s", char_type);
  Rprintf("\r");
  R_FlushConsole();
  R_CheckUserInterrupt();
}
#endif

typedef double tnumeric;

struct PathElem {
  PathElem(int d, bool o, tnumeric z, tnumeric w) : d(d), o(o), z(z), w(w) {}
  int d;
  bool o;
  tnumeric z, w;
};

typedef std::vector<PathElem> Path;

void extend(Path &m, tnumeric p_z, bool p_o, int p_i) {
  int depth = m.size();

  PathElem tmp(p_i, p_o, p_z, (depth == 0) ? 1.0 : 0.0);

  m.push_back(tmp);

  for (int i = depth - 1; i >= 0; i--) {
    m[i + 1].w += p_o * m[i].w * (i + 1) / static_cast<tnumeric>(depth + 1);
    m[i].w = p_z * m[i].w * (depth - i) / static_cast<tnumeric>(depth + 1);
  }
}

void unwind(Path &m, int i) {
  int depth = m.size() - 1;
  tnumeric n = m[depth].w;

  if (m[i].o != 0) {
    for (int j = depth - 1; j >= 0; --j) {
      tnumeric tmp = m[j].w;
      m[j].w = n * (depth + 1) / static_cast<tnumeric>(j + 1);
      n = tmp - m[j].w * m[i].z * (depth - j) / static_cast<tnumeric>(depth + 1);
    }
  } else {
    for (int j = depth - 1; j >= 0; --j) {
      m[j].w = (m[j].w * (depth + 1)) / static_cast<tnumeric>(m[i].z * (depth - j));
    }
  }

  for (int j = i; j < depth; ++j) {
    m[j].d = m[j + 1].d;
    m[j].z = m[j + 1].z;
    m[j].o = m[j + 1].o;
  }

  m.pop_back();
}

tnumeric unwound_sum(const Path &m, int i) {
  int depth = m.size() - 1;
  tnumeric total = 0;

  if (m[i].o != 0) {
    tnumeric n = m[depth].w;
    for (int j = depth - 1; j >= 0; --j) {
      tnumeric tmp = n / static_cast<tnumeric>((j + 1) * m[i].o);
      total += tmp;
      n = m[j].w - tmp * m[i].z * (depth - j);
    }
  } else {
    for (int j = depth - 1; j >= 0; --j) {
      total += m[j].w / static_cast<tnumeric>((depth - j) * m[i].z);
    }
  }

  return total * (depth + 1);
}

// SHAP computation for a single decision tree
void recurse(const IntegerVector &yes, const IntegerVector &no, const IntegerVector &missing, const IntegerVector &feature,
             const LogicalVector &is_leaf, const NumericVector &value, const NumericVector &cover,
             const NumericVector &split, const IntegerVector &decision_type, const NumericVector &observation, const LogicalVector &observation_is_na,
             NumericVector &shaps, Path &m, int j, tnumeric p_z, bool p_o, int p_i,
             int condition, int condition_feature, tnumeric condition_fraction) {

  if (condition_fraction == 0) {
    return;
  }

  if (p_z == 0) { // entering a node with Cover = 0
    return;
  }

  if (condition == 0 || // not calculating interactions
      condition_feature != p_i) {
    extend(m, p_z, p_o, p_i);
  }

  if (is_leaf[j]) {
    for (int i = 1; i < m.size(); ++i) {
      shaps[m[i].d] += unwound_sum(m, i) * (m[i].o - m[i].z) * condition_fraction * value[j];
    }
  } else {
    tnumeric i_z = 1.0;
    bool i_o = 1;

    // undo previous extension if we have already seen this feature
    for (int k = 1; k < m.size(); ++k) {
      if (m[k].d == feature[j]) {
        i_z = m[k].z;
        i_o = m[k].o;
        unwind(m, k);
        break;
      }
    }

    if ((missing[j] == NA_INTEGER) || (missing[j] == no[j]) || (missing[j] == yes[j])) { //'missing' is one of ['yes', 'no'] nodes, or is NA
      int hot = no[j];

      if (observation_is_na[feature[j]]) {
        hot = missing[j];
      } else if (((decision_type[j] == 1) && (observation[feature[j]] <= split[j]))
                   || ((decision_type[j] == 2) && (observation[feature[j]] < split[j]))) {
        hot = yes[j];
      }
      int cold = (hot == yes[j]) ? no[j] : yes[j];

      // divide up the condition_fraction among the recursive calls
      // if we are not calculating interactions then condition fraction is always 1
      tnumeric hot_condition_fraction = condition_fraction;
      tnumeric cold_condition_fraction = condition_fraction;
      if (feature[j] == condition_feature) {
        if (condition > 0) {
          cold_condition_fraction = 0;
        } else if (condition < 0) {
          hot_condition_fraction *= cover[hot] / static_cast<tnumeric>(cover[j]);
          cold_condition_fraction *= cover[cold] / static_cast<tnumeric>(cover[j]);
        }
      }

      Path m_copy = Path(m);
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps,
              m, hot,
              i_z * cover[hot] / static_cast<tnumeric>(cover[j]),
              i_o,
              feature[j],
              condition, condition_feature, hot_condition_fraction);
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps,
              m_copy, cold,
              i_z * cover[cold] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j],
              condition, condition_feature, cold_condition_fraction);
    } else { // 'missing' node is a third son = not one of ['yes', 'no'] nodes
      int hot = no[j];
      int cold1 = yes[j];
      int cold2 = missing[j];
      if (observation_is_na[feature[j]]) {
        hot = missing[j];
        cold1 = yes[j];
        cold2 = no[j];
      } else if(((decision_type[j] == 1) && (observation[feature[j]] <= split[j]))
                  || ((decision_type[j] == 2) && (observation[feature[j]] < split[j]))) {
        hot = yes[j];
        cold1 = missing[j];
        cold2 = no[j];
      }

      // divide up the condition_fraction among the recursive calls
      // if we are not calculating interactions condition fraction is always 1
      tnumeric hot_condition_fraction = condition_fraction;
      tnumeric cold1_condition_fraction = condition_fraction;
      tnumeric cold2_condition_fraction = condition_fraction;
      if (feature[j] == condition_feature) {
        if (condition > 0) {
          cold1_condition_fraction = 0;
          cold2_condition_fraction = 0;
        } else if (condition < 0) {
          hot_condition_fraction *= cover[hot] / static_cast<tnumeric>(cover[j]);
          cold1_condition_fraction *= cover[cold1] / static_cast<tnumeric>(cover[j]);
          cold2_condition_fraction *= cover[cold2] / static_cast<tnumeric>(cover[j]);
        }
      }

      Path m_copy1 = Path(m);
      Path m_copy2 = Path(m);
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps,
              m, hot,
              i_z * cover[hot] / static_cast<tnumeric>(cover[j]),
              i_o,
              feature[j],
              condition, condition_feature, hot_condition_fraction);
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps,
              m_copy1, cold1,
              i_z * cover[cold1] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j],
              condition, condition_feature, cold1_condition_fraction);
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps,
              m_copy2, cold2,
              i_z * cover[cold2] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j],
              condition, condition_feature, cold2_condition_fraction);
    }
  }
}

// [[Rcpp::export]]
NumericVector treeshap_cpp(DataFrame x, DataFrame is_na, IntegerVector roots,
                             IntegerVector yes, IntegerVector no, IntegerVector missing, IntegerVector feature,
                             NumericVector split, IntegerVector decision_type,
                             LogicalVector is_leaf, NumericVector value, NumericVector cover,
                             bool verbose) {
  NumericMatrix shaps(x.ncol(), x.nrow());

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
  if (verbose) {
    initProgressBar();
  }
#endif

  for (int obs = 0; obs < x.ncol(); obs++) {
    NumericVector observation = x[obs];
    LogicalVector observation_is_na = is_na[obs];

    NumericVector shaps_row(x.nrow());

    for (int i = 0; i < roots.size(); ++i) {
      Path m;
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps_row,
              m, roots[i], 1, 1, -1,
              0, 0, 1);
    }

    shaps(obs, _) = shaps_row;

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
    if (verbose) {
      updateProgressBar(obs + 1, x.ncol());
    }
#endif
  }

  return shaps;
}


// recursive tree traversal listing all features in the tree
void unique_features_tree_traversal(int node, const IntegerVector &yes, const IntegerVector &no,
                                    const IntegerVector &missing, const IntegerVector &feature, const LogicalVector &is_leaf,
                                    std::vector<int> &tree_features) {
  if (!is_leaf[node]) {
    tree_features.push_back(feature[node]);
    unique_features_tree_traversal(yes[node], yes, no, missing, feature, is_leaf, tree_features);
    unique_features_tree_traversal(no[node], yes, no, missing, feature, is_leaf, tree_features);
    if (missing[node] != NA_INTEGER && missing[node] != yes[node] && missing[node] != no[node]) {
      unique_features_tree_traversal(missing[node], yes, no, missing, feature, is_leaf, tree_features);
    }
  }
}

// function listing all unique features inside the tree
std::vector<int> unique_features(int root, const IntegerVector &yes, const IntegerVector &no,
                                 const IntegerVector &missing, const IntegerVector &feature, const LogicalVector &is_leaf) {
  std::vector<int> tree_features;
  unique_features_tree_traversal(root, yes, no, missing, feature, is_leaf, tree_features);

  // removing duplicates
  std::sort(tree_features.begin(), tree_features.end());
  auto last = std::unique(tree_features.begin(), tree_features.end());
  tree_features.erase(last, tree_features.end());

  return tree_features;
}

// [[Rcpp::export]]
List treeshap_interactions_cpp(DataFrame x, DataFrame is_na, IntegerVector roots,
                           IntegerVector yes, IntegerVector no, IntegerVector missing, IntegerVector feature,
                           NumericVector split, IntegerVector decision_type,
                           LogicalVector is_leaf, NumericVector value, NumericVector cover,
                           bool verbose) {
  NumericMatrix shaps(x.ncol(), x.nrow());
  NumericVector interactions(x.ncol() * x.nrow() * x.nrow());


#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
  if (verbose) {
    initProgressBar();
  }
#endif

  for (int obs = 0; obs < x.ncol(); obs++) {
    NumericVector observation = x[obs];
    LogicalVector observation_is_na = is_na[obs];

    NumericMatrix interactions_slice(x.nrow(), x.nrow());
    NumericVector shaps_row(x.nrow());
    NumericVector diagonal(x.nrow());

    for (int i = 0; i < roots.size(); ++i) {
      Path m;
      recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, shaps_row,
              m, roots[i], 1, 1, -1,
              0, 0, 1); // standard shaps computation

      std::vector<int> tree_features = unique_features(roots[i], yes, no, missing, feature, is_leaf);
      for (auto tree_feature : tree_features) {
        NumericVector with(x.nrow());
        NumericVector without(x.nrow());

        Path m_with;
        recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, with,
                m_with, roots[i], 1, 1, -1,
                1, tree_feature, 1);
        Path m_without;
        recurse(yes, no, missing, feature, is_leaf, value, cover, split, decision_type, observation, observation_is_na, without,
                m_without, roots[i], 1, 1, -1,
                -1, tree_feature, 1);

        NumericVector v = (with - without) / 2;
        interactions_slice(tree_feature, _) = interactions_slice(tree_feature, _) + v;
        diagonal = diagonal - v;
      }
    }

    // filling diagonal
    diagonal = shaps_row + diagonal;
    for (int k = 0; k < x.nrow(); ++k) {
      interactions_slice(k, k) = diagonal[k];
    }

    // prescribing results from observation's vector and matrix to result's matrix and "array"
    shaps(obs, _) = shaps_row;
    for (int i = 0; i < x.nrow(); i++) {
      for (int j = 0; j < x.nrow(); j++) {
        interactions[obs * x.nrow() * x.nrow() + i * x.nrow() + j] = interactions_slice(i, j);
      }
    }

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
    if (verbose) {
      updateProgressBar(obs + 1, x.ncol());
    }
#endif
  }

  List ret = List::create(Named("shaps") = shaps, _["interactions"] = interactions);
  return ret;
}
