#ifndef YZBOOST_CSV_H_
#define YZBOOST_CSV_H_

#include <vector>
#include <string>

#include "common.h"
#include "math.h"

namespace yzboost {
namespace csv {

bool read_csv(const std::string& filename, Matrix& data,
              size_t buff_size=(1<<22), size_t max_line_len=10240);

}  // csv
}  // yzboost

#endif
