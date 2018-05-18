#ifndef YZBOOST_UTIL_H_
#define YZBOOST_UTIL_H_

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "math.h"

namespace yzboost {
namespace util {

typedef std::chrono::high_resolution_clock::time_point T;

T now();
double seconds_since(T since);
double seconds(T start, T finish);

Matrix extract_validation_split(Matrix& src, double fraction, PRNG& prng=yzboost::default_prng);
Vector extract_column(Matrix& src, int column);

void diagnose_openmp();

}  // util
}  // yzboost

#endif
