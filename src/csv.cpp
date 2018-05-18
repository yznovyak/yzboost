#include "csv.h"

#include <cctype>
#include <cstdio>

#include <string>
#include <cstring>
#include <memory>

#include "log.h"

using std::unique_ptr;
using std::vector;
using std::string;
using std::istream;

namespace yzboost {
namespace csv {

void parse_line(char* line, Vector& dst) {
  // replace commas with spaces.  also deal with comments -- delete
  // them and everything afterwards
  for (char* p = line; *p; p++) {
    if (*p == '#') {
      *p = 0;
      break;
    }
    if (*p == ',')
      *p = ' ';
  }

  for (char* p = line; *p; ) {
    while (*p && isspace(*p)) ++p;  // skip spaces
    if (!*p)
      break;  // done reading line

    dst.emplace_back(atof(p));
    while (*p && !isspace(*p)) ++p;  // move over
  }
}

void process_lines(const vector<char*>& lines, Matrix& data) {
  size_t offset = data.size();
  data.resize(data.size() + lines.size());
  #pragma omp parallel for
  for (size_t i = 0; i < lines.size(); i++)
    parse_line(lines[i], data[offset+i]);
}

bool read_csv(const string& filename, Matrix& data, size_t buff_size, size_t max_line_len) {
  FILE* f = fopen(filename.c_str(), "r");
  if (!f)
    return false;

  unique_ptr<char[]> line(new char[buff_size]);
  vector<char*> line_starts;
  data.clear();

  while (true) {
    line_starts.clear();
    char* p = line.get();
    while (buff_size - (p - line.get()) >= max_line_len) {
      if (!fgets(p, 10240, f))
        break;

      line_starts.push_back(p);
      p += strlen(p) + 2;
    }
    if (line_starts.empty())
      break;

    process_lines(line_starts, data);
  }

  size_t next_pos = 0;
  for (size_t i = 0; i < data.size(); i++)
    if (!data[i].empty())
      data[next_pos++] = data[i];
  data.resize(next_pos);

  for (size_t i = 0; i < data.size(); i++)
    if (data[i].size() != data[0].size()) {
      LOG << "CSV file provided non-rectangular matrix.";
      return false;
    }

  return true;
}

}  // csv
}  // yzboost
