#include "classifier.h"

#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <mutex>
#include <cassert>

#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <iomanip>

#include "util.h"
#include "math.h"
#include "log.h"

using std::accumulate;
using std::max_element;
using std::pair;
using std::istream;
using std::ostream;
using std::vector;
using std::string;
using std::uniform_int_distribution;

using yzboost::math::l1_norm;
using yzboost::math::logit;
using yzboost::math::mean;
using yzboost::math::safe_logloss;
using yzboost::math::sigmoid;
using yzboost::util::seconds;
using yzboost::util::seconds_since;
using yzboost::util::now;
using yzboost::util::T;

namespace yzboost {

inline float loss(bool y_true, float x) {
  if (!y_true) x = -x;
  if (x >= 15) return 0;
  if (x <= -15) return -x;
  return std::log(exp(-x) + 1);
}

inline float grad(bool y_true, float x) {
  if (y_true) {
    if (x >= 15) return 0;
    if (x <= -15) return -1;
    return -1.f / (exp(x) + 1);
  } else
    return -grad(!y_true, -x);
}

namespace {  // discretization stuff

std::mutex precompute_mutex;
bool is_already_precomputed = false;

const int kMult = 100000;
const float kMaxAbsYf = 100;
const int kLastIndex = kMaxAbsYf*2*kMult;

vector<float> loss_at[2], grad_at[2];

inline float mean_loss(const vector<int>& y_true, const vector<int>& yF, int shift) {
  double sum = 0;
  #pragma omp simd
  for (size_t i = 0; i < yF.size(); i++) {
    int p = yF[i] + shift;
    if (p < 0) p = 0;
    if (p > kLastIndex)
      p = kLastIndex;
    sum += loss_at[y_true[i]][p];
  }
  return sum / yF.size();
}

int float_to_int(float x) {
  if (x < -kMaxAbsYf) x = -kMaxAbsYf;
  if (x > kMaxAbsYf) x = kMaxAbsYf;
  return (x + kMaxAbsYf) * kMult;
}

float int_to_float(int i) {
  return float(i - kMult*kMaxAbsYf)/kMult;
}

void precompute() {
  std::lock_guard<std::mutex> guard(precompute_mutex);
  if (is_already_precomputed)
    return;

  for (int ytrue = 0; ytrue <= 1; ytrue++) {
    loss_at[ytrue].resize(kLastIndex+1);
    grad_at[ytrue].resize(kLastIndex+1);
  }
  #pragma omp parallel for
  for (int i = 0; i <= kLastIndex; i++) {
    float p = int_to_float(i);
    loss_at[0][i] = loss(false, p);
    grad_at[0][i] = grad(false, p);
    loss_at[1][i] = loss(true, p);
    grad_at[1][i] = grad(true, p);
  }
}

Vector compute_bin_dividers(Vector& sample, int max_bins) {
  if (max_bins == 1)
    return {};  // ha-ha

  sort(sample.begin(), sample.end());

  Vector dividers;
  for (int i = 0; i < max_bins-1; i++) {
    float x = sample[(i+1)*sample.size() / max_bins];
    if (dividers.empty() || dividers.back() != x)
      dividers.emplace_back(x);
  }

  if (dividers.back() == *max_element(sample.begin(), sample.end()))
    dividers.pop_back();  // otherwise the last bucket will be empty

  return dividers;
}

inline int compute_bin(const Vector& dividers, float x) {
  return lower_bound(dividers.begin(), dividers.end(), x) - dividers.begin();
}

}  // namespace (anonymous)

YZBoostClassifier::YZBoostClassifier()
    : bias_(std::numeric_limits<float>::quiet_NaN()) {
}

void YZBoostClassifier::fit(const Matrix& X, const Vector& y) {
  LOG << "fit() was invoked with following options:"
      << " --num_trees=" << opts.num_trees
      << " --tree_depth=" << opts.tree_depth;
  assert(!X.empty());
  assert(X.size() == y.size());
  num_features_ = X[0].size();
  T t0 = now();

  precompute();
  discretize(X, y);
  update_bias();
  Vector yF = eval_trees(X);

  LOG << "Misc preparations took " << seconds_since(t0) << " sec";
  init_stats();
  for (int i = 0; i < opts.num_trees; i++) {
    vector<int> int_yF(yF.size());
    for (size_t i = 0; i < yF.size(); i++) {
      int_yF[i] = float_to_int(yF[i]);
      if (yF[i] < -kMaxAbsYf*0.8 || yF[i] > kMaxAbsYf*.8) {
        LOG << "Variables diverged way out of reasonable bounds -- stopping training due to possible overfit";
        return;
      }
    }

    trees_.emplace_back(make_tree(int_yF));
    yF += trees_.back().eval(X);

    log_stats(y, yF);
  }
}

void YZBoostClassifier::discretize(const Matrix& X, const Vector& y) {
  int N = X.size(), F = X[0].size(), B = opts.discretization_max_bins;
  uniform_int_distribution<int> rnd(0, N-1);

  y_true_ = vector<int>(N);
  for (int i = 0; i < N; i++)
    y_true_[i] = (y[i] > 0.5);

  X_bin_.assign(F, vector<int>(N));
  bin_dividers_.resize(F);

  #pragma omp parallel for
  for (int f = 0; f < F; f++) {
    // Sample features
    Vector feature_samples(opts.discretization_num_subsamples);
    for (int i = 0; i < opts.discretization_num_subsamples; i++)
      feature_samples[i] = X[rnd(*opts.prng)][f];

    // Compute divisors between different bins
    bin_dividers_[f] = compute_bin_dividers(feature_samples, B);

    // Compute bin for every feature
    for (int i = 0; i < N; i++)
      X_bin_[f][i] = compute_bin(bin_dividers_[f], X[i][f]);
  }
}

void YZBoostClassifier::update_bias() {
  // init bias as logit(mean(y)), because it minimizes logloss
  if (std::isnan(bias_)) {
    int num_true = count(y_true_.begin(), y_true_.end(), true);
    bias_ = logit(float(num_true) / y_true_.size());
  }
}

double choose_leaf_values(const vector<int>& y_true, const vector<int>& yF) {
  if (y_true.empty())
    return 0;

  const static double kGoldenRatio = (sqrt(5.) + 1) / 2;
  const static int kTolerance = kMult / 10000;  // approximately 10^-4

  int a = -10*kMult, b = +10*kMult;

  int c = b - (b - a) / kGoldenRatio;
  int d = a + (b - a) / kGoldenRatio;
  while (abs(c - d) > kTolerance) {
    if (mean_loss(y_true, yF, c) < mean_loss(y_true, yF, d))
      b = d; else
      a = c;

    c = b - (b - a) / kGoldenRatio;
    d = a + (b - a) / kGoldenRatio;
  }
  return double((a + b) / 2) / kMult;
}

void YZBoostClassifier::optimize_tree(Tree& tree, const vector<int>& sample_to_leaf,
                                      const vector<int>& yF) const {
  int N = sample_to_leaf.size(), L = 1 << tree.depth();

  vector<vector<int>> leaf_yF(L);
  vector<vector<int>> leaf_y_true(L);
  for (int i = 0; i < N; i++) {
    int j = sample_to_leaf[i];
    leaf_y_true[j].push_back(y_true_[i]);
    leaf_yF[j].emplace_back(yF[i]);
  }

  tree.leaf_values.resize(L);

  #pragma omp parallel for
  for (int l = 0; l < L; l++)
    tree.leaf_values[l] = choose_leaf_values(leaf_y_true[l], leaf_yF[l]);
}

Tree YZBoostClassifier::make_tree(const vector<int>& yF) const {
  Vector gain(yF.size());
  #pragma omp parallel for
  for (size_t i = 0; i < yF.size(); i++)
    gain[i] = grad_at[y_true_[i]][yF[i]];
  pair<Tree, vector<int>> tree_and_index = grow_tree(gain);
  optimize_tree(tree_and_index.first, tree_and_index.second, yF);
  return tree_and_index.first;
}

pair<Tree, vector<int>> YZBoostClassifier::grow_tree(const Vector& gain) const {
  Tree tree;
  int N = gain.size(), F = X_bin_.size();
  vector<int> sample_to_leaf(N);

  // At level L all training samples are assigned to one of 2^L
  // partitions.  Each partition contains training samples that answer
  // identically to the first L questions of the ODT.
  for (int level = 0; level < opts.tree_depth; level++) {
    int P = 1 << level;  // number of partitions we currently have

    // split_total_gain[f][i] is the "gain" for splitting dataset at
    // this level by rule X[f] <= bin_dividers_[f,i].
    Matrix split_total_gain = make_matrix(F, opts.discretization_max_bins);
    #pragma omp parallel for
    for (int f = 0; f < F; f++) {  // try spliting at F
      int B = bin_dividers_[f].size() + 1;  // number of bins
      split_total_gain[f].resize(B-1);
      if (B <= 1) continue;

      Matrix L = make_matrix(B, P);

      // !!!!! two most time consuming lines below :)
      for (int i = 0; i < N; i++)
        L[X_bin_[f][i]][sample_to_leaf[i]] += gain[i];

      for (int i = 1; i < B; i++)
        L[i] += L[i-1];


      // compute gain for all thresholds
      // TODO(yznovyak): some better way to define gain?
      for (int t = 0; t < B-1; t++)
        split_total_gain[f][t] += l1_norm(L[t]) + l1_norm(L[B-1], L[t]);
    }

    // Find best split
    float best_gain = std::numeric_limits<float>::lowest();
    int best_feature = -1, best_split_ind = -1;
    for (int f = 0; f < F; f++) {
      for (size_t i = 0; i < split_total_gain[f].size(); i++) {
        if (best_gain < split_total_gain[f][i]) {
          best_gain = split_total_gain[f][i];
          best_feature = f;
          best_split_ind = i;
        }
      }
    }
    if (best_feature == -1)  // doesn't make sense to split further...
      break;

    // Grow tree and recompute part[]
    tree.splits.push_back(Tree::Split{best_feature, bin_dividers_[best_feature][best_split_ind]});
    for (int i = 0; i < N; i++) {
      sample_to_leaf[i] <<= 1;
      sample_to_leaf[i] |= (X_bin_[best_feature][i] <= best_split_ind ? 0 : 1);
    }
  }

  return {tree, sample_to_leaf};
}

void YZBoostClassifier::init_stats() {
  if (!yzboost::logging::FLAG_logging_enabled)
    return;

  ts_train_accuracy_ = 0;
  ts_train_loss_ = 0;
  ts_test_accuracy_ = 0;
  ts_test_loss_ = 0;
  timestamp_ = now();
}

string stats_str(double acc_now, double acc_prev, double loss_now, double loss_prev) {
  char msg[128];
  sprintf(msg, "acc=%.6lf (%+.2e)  loss=%.6lf (%+.2e)", acc_now, acc_now-acc_prev, loss_now, loss_now-loss_prev);
  return msg;
}

pair<double, double> measure(const Vector& y_true, const Vector& y_pred) {
  int num_ok = 0;
  double sum_loss = 0;
  for (size_t i = 0; i < y_true.size(); i++) {
    bool p = y_true[i] > 0.5, q = y_pred[i] > 0.5;
    if (p == q)
      num_ok++;
    sum_loss += safe_logloss(y_true[i], y_pred[i]);
  }
  return {double(num_ok) / y_true.size(), sum_loss / y_true.size()};
}

void YZBoostClassifier::log_stats(const Vector& y_true, const Vector& yF) {
  if (!yzboost::logging::FLAG_logging_enabled)
    return;

  double duration = seconds_since(timestamp_);
  timestamp_ = now();


  Vector y_pred(yF.size());
  for (size_t i = 0; i < yF.size(); i++)
    y_pred[i] = sigmoid(yF[i]);
  pair<double, double> m = measure(y_true, y_pred);

  string msg = "train: " + stats_str(m.first, ts_train_accuracy_, m.second, ts_train_loss_);
  ts_train_accuracy_ = m.first;
  ts_train_loss_ = m.second;

  if (opts.valid_X) {
    y_pred = predict_proba(*opts.valid_X);
    pair<double, double> m = measure(*opts.valid_y, y_pred);
    msg += "      test: " + stats_str(m.first, ts_test_accuracy_, m.second, ts_test_loss_);
    ts_test_accuracy_ = m.first;
    ts_test_loss_ = m.second;
  }

  LOG << std::setw(3) << trees_.size() << " " << msg << "      @ " << duration << " sec";
  timestamp_ = now();
}


float YZBoostClassifier::eval_trees(const Vector& x) const {
  float res = bias_;
  for (const Tree& t : trees_)
    res += t.eval(x);
  return res;
}

Vector YZBoostClassifier::eval_trees(const Matrix& X) const {
  Vector res(X.size());
  #pragma omp parallel for
  for (size_t i = 0; i < X.size(); i++)
    res[i] = eval_trees(X[i]);
  return res;
}

int YZBoostClassifier::predict(const Vector& x) const {
  assert(x.size() == size_t(num_features_));
  return eval_trees(x) >= 0 ? 1 : 0;
}

vector<int> YZBoostClassifier::predict(const Matrix& X) const {
  vector<int> res(X.size());
  for (size_t i = 0; i < X.size(); i++)
    res[i] = predict(X[i]);
  return res;
}

float YZBoostClassifier::predict_proba(const Vector& x) const {
  assert(x.size() == size_t(num_features_));
  return sigmoid(eval_trees(x));
}

Vector YZBoostClassifier::predict_proba(const Matrix& X) const {
  Vector res(X.size());
  for (size_t i = 0; i < X.size(); i++)
    res[i] = predict_proba(X[i]);
  return res;
}

// HACK BECAUSE I'M RUNNING OUT OF TIME
static_assert(sizeof(float) == sizeof(int32_t), "hack ... sorry");

uint32_t& as_u32(float& x) {  // i'm so fucking sorry...
  return *reinterpret_cast<uint32_t*>(&x);
}

void YZBoostClassifier::load(istream& is) {
  int num_trees;
  is >> num_trees >> as_u32(bias_) >> num_features_;
  trees_.resize(num_trees);
  for (Tree& t : trees_)
    t.load(is);
}

void YZBoostClassifier::save(ostream& os) {
  os << trees_.size() << " " << as_u32(bias_) << " " << num_features_ << "\n";
  for (auto& t : trees_)
    t.save(os);
}

float Tree::eval(const Vector& x) const {
  int ind = 0;
  for (const Split& s : splits)
    ind = (ind << 1) | (x[s.feature] <= s.threshold ? 0 : 1);
  return leaf_values[ind];
}

Vector Tree::eval(const Matrix& X) const {
  Vector res(X.size());
  for (size_t i = 0; i < X.size(); i++)
    res[i] = eval(X[i]);
  return res;
}

void Tree::save(ostream& os) {
  os << depth() << "\n";
  for (Split s : splits)
    os << s.feature << " " << as_u32(s.threshold) << "\n";
  for (float f : leaf_values)
    os << f << " ";
  os << "\n";
}

void Tree::load(istream& is) {
  int depth;
  is >> depth;
  splits.resize(depth);
  leaf_values.resize(1 << depth);
  for (Split& s : splits)
    is >> s.feature >> as_u32(s.threshold);
  for (float& f : leaf_values)
    is >> f;
}

}  // yzboost
