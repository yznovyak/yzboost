#include "util.h"

#include <algorithm>
#include <vector>
#include <chrono>
#include <random>

#include "log.h"

using namespace std;


using std::lower_bound;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;

using std::uniform_real_distribution;

namespace yzboost {
namespace util {

T now() {
  return std::chrono::high_resolution_clock::now();
}

double seconds(T start, T end) {
  return duration_cast<duration<double>>(end - start).count();
}

double seconds_since(T since) {
  return seconds(since, now());
}

Matrix extract_validation_split(Matrix& src, double fraction, PRNG& prng) {
  std::uniform_real_distribution<double> rnd(0, 1);
  size_t N = 0;
  Matrix dst;
  for (size_t i = 0; i < src.size(); i++) {
    if (rnd(prng) < fraction)
      dst.emplace_back(src[i]); else
      src[N++] = src[i];
  }
  src.resize(N);
  return dst;
}

Vector extract_column(Matrix& src, int column) {
  Vector vec(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    vec[i] = src[i][column];
    src[i].erase(src[i].begin() + column);
  }
  return vec;
}

void diagnose_openmp() {
#ifdef _OPENMP
  LOG << "OpenMP present";
#else
  LOG << "No OpenMP :(";
#endif
}


}  // util
}  // yzboost
