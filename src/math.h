#ifndef YZBOOST_MATH_H_
#define YZBOOST_MATH_H_

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace yzboost {

typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;

inline std::vector<std::vector<float>> make_matrix(int rows, int cols, float initial_value = 0) {
  return std::vector<std::vector<float>>(rows, std::vector<float>(cols, initial_value));
}

inline void operator+=(Vector& a, const Vector& b) {
  #pragma omp simd
  for (size_t i = 0; i < a.size(); i++)
    a[i] += b[i];
}

namespace math {

inline double sigmoid(double x) {
  x = exp(x);
  return x / (x + 1);
}

inline double logit(double p) {
  return -std::log(1/p - 1);
}

inline double safe_logloss(double y_true, double y_pred, double eps = 1e-15) {
  if (y_true <= 0.5)
    y_pred = 1-y_pred;
  return -std::log(fmax(eps, y_pred));
}

inline float mean(const Vector& vec) {
  // double is intentional
  return std::accumulate(vec.begin(), vec.end(), double(0)) / vec.size();
}

inline float l1_norm(const Vector& v) {  // double is intentional
  float sum = 0;
  #pragma omp simd reduction(+:sum)
  for (size_t i = 0; i < v.size(); i++)
    sum += (v[i] < 0 ? -v[i] : v[i]);
  return sum;
}

inline float l1_norm(const Vector& a, const Vector& b) {
  int n = a.size();
  float sum = 0;
  #pragma omp simd reduction(+:sum)
  for (int i = 0; i < n; i++)
    sum += abs(a[i] - b[i]);
  return sum;
}

}  // math
}  // yzboost

#endif
