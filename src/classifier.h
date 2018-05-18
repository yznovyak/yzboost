#ifndef YZBOOST_CLASSIFIER_H_
#define YZBOOST_CLASSIFIER_H_

#include <vector>
#include <istream>
#include <ostream>

#include "common.h"
#include "math.h"
#include "util.h"

namespace yzboost {

struct Tree {  // An oblivious decision tree
  struct Split {
    int feature;
    float threshold;
  };

  int depth() const { return splits.size(); }

  float eval(const Vector& x) const;
  Vector eval(const Matrix& X) const;

  void save(std::ostream& os);
  void load(std::istream& os);

  std::vector<Split> splits;
  Vector leaf_values;
};

class YZBoostClassifier {
 public:
  YZBoostClassifier();

  struct FitOptions {  // how to do fit()
    int num_trees = 10;
    int tree_depth = 10;

    int discretization_max_bins = 100;
    int discretization_num_subsamples = 50000;

    Matrix* valid_X = nullptr;
    Vector* valid_y = nullptr;

    PRNG* prng = &yzboost::default_prng;
  };

  FitOptions opts;

  void fit(const Matrix& X, const Vector& y);
  int predict(const Vector& x) const;
  std::vector<int> predict(const Matrix& X) const;
  float predict_proba(const Vector& x) const;
  Vector predict_proba(const Matrix& X) const;
  void save(std::ostream& os);
  void load(std::istream& is);

  float get_bias() const { return bias_; }
  int get_num_features() const { return num_features_; }
  std::vector<Tree> get_trees() const { return trees_; }
 private:
  void update_bias();
  void discretize(const Matrix& X, const Vector& y);
  float eval_trees(const Vector& X) const;
  Vector eval_trees(const Matrix& X) const;
  Tree make_tree(const std::vector<int>& y_F) const;
  void optimize_tree(Tree& tree, const std::vector<int>& sample_to_leaf,
                     const std::vector<int>& y_F) const;
  std::pair<Tree, std::vector<int>> grow_tree(const Vector& gain) const;

  void init_stats();
  void log_stats(const Vector& y_true, const Vector& yF);

  int num_features_;
  float bias_;
  std::vector<int> y_true_;
  std::vector<Tree> trees_;

  std::vector<std::vector<int>> X_bin_;  // [feature][sample] -> bin
  Matrix bin_dividers_;  // bin_dividers_[feature][i] is the threshold between feature's bins i and i+1.

  yzboost::util::T timestamp_;
  float ts_train_accuracy_, ts_train_loss_;
  float ts_test_accuracy_, ts_test_loss_;

};

}  // yzboost

#endif
