#ifndef YZBOOST_LOG_H_
#define YZBOOST_LOG_H_

#include <iostream>
#include <vector>
#include <utility>
#include <ostream>

namespace yzboost {

template <class T1, class T2>
std::ostream& operator <<(std::ostream& os, const std::pair<T1, T2>& v) {
  os << "(" << v.first << " " << v.second << ")";
  return os;
}

template <typename T,
          typename U = typename T::value_type,
          typename = typename std::enable_if<!std::is_same<T, std::string>::value>::type>
std::ostream& operator<<(std::ostream& os, const T& v) {
  os << '[';
  for (const auto& it : v)
    os << it << " ";
  if (!v.empty()) os << "\b";
  os << "]";
  return os;
}

namespace logging {

struct NullStream: public std::ostream {
  NullStream() : std::ostream(0) {}
};

template <typename T>
NullStream& operator<<(NullStream& s, const T&) { return s; }

extern NullStream null_stream;

extern bool FLAG_show_file;
extern bool FLAG_logging_enabled;

class Log {
 public:
  Log(const char* c_filename, int line_num)
      : out_stream(FLAG_logging_enabled ? std::cout : null_stream) {
    if (FLAG_logging_enabled && FLAG_show_file) {
      std::string filename(c_filename);
      std::string basename = filename.substr(1 + filename.rfind('/'));

      int len_line_num = 0;
      for (int x = line_num; x; x /= 10)
        len_line_num++;

      pad(basename.length(), 16);
      out_stream << basename << ":" << line_num;
      pad(len_line_num, 4);
    }
  }
  ~Log() {
    stream() << std::endl;
  }

  inline std::ostream& stream() {
    return out_stream;
  }

 private:
    std::ostream& out_stream;

    void pad(int have, int need) {
      for (int i = have; i < need; i++)
        out_stream << ' ';
    }
};

}  // logging
}  // yzboost

#define LOG yzboost::logging::Log(__FILE__, __LINE__).stream()

#endif
