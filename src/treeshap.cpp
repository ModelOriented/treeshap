#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

typedef double tnumeric;

struct PathElem {
  unsigned d;
  bool o;
  tnumeric z, w;
};

typedef std::vector<PathElem> Path;

void extend(Path &m, tnumeric p_z, bool p_o, unsigned p_i) {
  unsigned depth = m.size();

  PathElem tmp = {.d = p_i, .o = p_o, .z = p_z, .w = (depth == 0) ? 1.0 : 0.0};

  m.push_back(tmp);

  for (int i = depth - 1; i >= 0; i--) {
    m[i + 1].w += p_o * m[i].w * (i + 1) / static_cast<tnumeric>(depth + 1);
    m[i].w = p_z * m[i].w * (depth - i) / static_cast<tnumeric>(depth + 1);
  }
}

void unwind(Path &m, unsigned i) {
  unsigned depth = m.size() - 1;
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

tnumeric unwound_sum(const Path &m, unsigned i) {
  unsigned depth = m.size() - 1;
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
             const LogicalVector &is_leaf, const NumericVector &value, const NumericVector &cover, const LogicalVector &fulfills,
             NumericVector &shaps, Path &m, unsigned j, tnumeric p_z, bool p_o, unsigned p_i) {
  extend(m, p_z, p_o, p_i);

  if (is_leaf[j]) {
    for (int i = 1; i < m.size(); ++i) {
      shaps[m[i].d] += unwound_sum(m, i) * (m[i].o - m[i].z) * value[j];
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

    if ((missing[j] == NA_INTEGER) | (missing[j] == no[j]) | (missing[j] == yes[j])) {
      unsigned hot = no[j];
      if (fulfills[j] == NA_LOGICAL) {
        hot = missing[j];
      } else if (fulfills[j]) {
        hot = yes[j];
      }
      unsigned cold = (hot == yes[j]) ? no[j] : yes[j];

      Path m_copy = Path(m);
      recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
              m, hot,
              i_z * cover[hot] / static_cast<tnumeric>(cover[j]),
              i_o,
              feature[j]);
      recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
              m_copy, cold,
              i_z * cover[cold] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j]);
    } else {
      unsigned hot = no[j];
      unsigned cold1 = yes[j];
      unsigned cold2 = missing[j];
      if (fulfills[j] == NA_LOGICAL) {
        hot = missing[j];
        cold1 = yes[j];
        cold2 = no[j];
      } else if (fulfills[j]) {
        hot = yes[j];
        cold1 = missing[j];
        cold2 = no[j];
      }

      Path m_copy1 = Path(m);
      Path m_copy2 = Path(m);
      recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
              m, hot,
              i_z * cover[hot] / static_cast<tnumeric>(cover[j]),
              i_o,
              feature[j]);
      recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
              m_copy1, cold1,
              i_z * cover[cold1] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j]);
      recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
              m_copy2, cold2,
              i_z * cover[cold2] / static_cast<tnumeric>(cover[j]),
              0,
              feature[j]);
    }
  }
}

// [[Rcpp::export]]
NumericVector treeshap_cpp(unsigned x_size, LogicalVector fulfills, IntegerVector roots,
                             IntegerVector yes, IntegerVector no, IntegerVector missing, IntegerVector feature,
                             LogicalVector is_leaf, NumericVector value, NumericVector cover) {
  NumericVector shaps(x_size);

  for (int i = 0; i < roots.size(); ++i) {
    Path m;
    recurse(yes, no, missing, feature, is_leaf, value, cover, fulfills, shaps,
            m, roots[i], 1, 1, -1);
  }

  return shaps;
}
