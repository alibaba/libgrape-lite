/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPE_IO_LINE_PARSER_BASE_H_
#define GRAPE_IO_LINE_PARSER_BASE_H_

#include <string>

namespace grape {

/**
 * @brief LineParserBase is the base class for line parsers.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 * @tparam OID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VDATA_T, typename EDATA_T>
class LineParserBase {
 public:
  LineParserBase() = default;

  virtual ~LineParserBase() = default;

  /**
   * @brief parser of edge file, to parse source vertex_id, destination
   * vertex_id from a line.
   *
   * @param line
   * @param u
   * @param v
   * @param e_data
   */
  virtual void LineParserForEFile(const std::string& line, OID_T& u, OID_T& v,
                                  EDATA_T& e_data) = 0;

  /**
   * @brief parser of vertex file, to parse vertex_id and vertex_data from a
   * string.
   *
   * @param line
   * @param u
   * @param v_data
   */
  virtual void LineParserForVFile(const std::string& line, OID_T& u,
                                  VDATA_T& v_data) = 0;
};

namespace internal {
// return the next position after char sequence that consumed by the matcher.
template <typename T>
inline const char* match(char const* str, T& r, char const* end = nullptr);

template <>
inline const char* match<int32_t>(char const* str, int32_t& r, char const*) {
  char* match_end;
  r = std::strtol(str, &match_end, 10);
  return match_end;
}

template <>
inline const char* match<int64_t>(char const* str, int64_t& r, char const*) {
  char* match_end;
  r = std::strtoll(str, &match_end, 10);
  return match_end;
}

template <>
inline const char* match<uint32_t>(char const* str, uint32_t& r, char const*) {
  char* match_end;
  r = std::strtoul(str, &match_end, 10);
  return match_end;
}

template <>
inline const char* match<uint64_t>(char const* str, uint64_t& r, char const*) {
  char* match_end;
  r = std::strtoull(str, &match_end, 10);
  return match_end;
}

template <>
inline const char* match<float>(char const* str, float& r, char const*) {
  char* match_end;
  r = std::strtof(str, &match_end);
  return match_end;
}

template <>
inline const char* match<double>(char const* str, double& r, char const*) {
  char* match_end;
  r = std::strtod(str, &match_end);
  return match_end;
}

template <>
inline const char* match<long double>(char const* str, long double& r,
                                      char const*) {
  char* match_end;
  r = std::strtold(str, &match_end);
  return match_end;
}

template <>
inline const char* match<std::string>(char const* str, std::string& r,
                                      char const* end) {
  int nlen1 = 0, nlen2 = 0;
  // skip preceding spaces or new line mark
  while (str + nlen1 != end && str[nlen1] &&
         (str[nlen1] == '\n' || str[nlen1] == ' ' || str[nlen1] == '\t')) {
    nlen1 += 1;
  }
  nlen2 = nlen1;
  while (str + nlen2 != end && str[nlen2] &&
         (str[nlen2] != '\n' && str[nlen2] != ' ' && str[nlen2] != '\t')) {
    nlen2 += 1;
  }
  r = std::string(str + nlen1, nlen2 - nlen1);
  return str + nlen2;
}

template <>
inline const char* match<grape::EmptyType>(char const* str, grape::EmptyType&,
                                           char const*) {
  return str;
}

}  // namespace internal

}  // namespace grape
#endif  // GRAPE_IO_LINE_PARSER_BASE_H_
