#ifndef YZBOOST_COMMON_H_
#define YZBOOST_COMMON_H_

#include <string>
#include <random>

namespace yzboost {

typedef std::mt19937 PRNG;
extern PRNG default_prng;

}  // yzboost

#endif
