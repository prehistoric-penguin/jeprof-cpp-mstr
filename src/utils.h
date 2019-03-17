#pragma once

#include <boost/utility/string_ref.hpp>
inline int stoi(boost::string_ref __str, size_t* __idx = 0, int __base = 10) {
  return __gnu_cxx::__stoa<long, int>(&std::strtol, "stoi", __str.data(), __idx,
                                      __base);
}

inline long stol(boost::string_ref __str, size_t* __idx = 0, int __base = 10) {
  return __gnu_cxx::__stoa(&std::strtol, "stol", __str.data(), __idx, __base);
}

inline unsigned long stoul(boost::string_ref __str, size_t* __idx = 0,
                           int __base = 10) {
  return __gnu_cxx::__stoa(&std::strtoul, "stoul", __str.data(), __idx, __base);
}

inline long long stoll(boost::string_ref __str, size_t* __idx = 0,
                       int __base = 10) {
  return __gnu_cxx::__stoa(&std::strtoll, "stoll", __str.data(), __idx, __base);
}

inline unsigned long long stoull(boost::string_ref __str, size_t* __idx = 0,
                                 int __base = 10) {
  return __gnu_cxx::__stoa(&std::strtoull, "stoull", __str.data(), __idx,
                           __base);
}
