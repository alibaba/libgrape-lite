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

#ifndef GRAPE_IO_TSV_LINE_PARSER_H_
#define GRAPE_IO_TSV_LINE_PARSER_H_

#include <regex>
#include <string>
#include <utility>

#include "grape/config.h"
#include "grape/io/line_parser_base.h"

namespace grape {

/**
 * @brief a default parser for tsv files.
 *
 * @tparam OID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VDATA_T, typename EDATA_T>
class TSVLineParser : public LineParserBase<OID_T, VDATA_T, EDATA_T> {
 public:
  TSVLineParser() {}

  virtual void LineParserForEFile(const std::string& line, OID_T& u, OID_T& v,
                                  EDATA_T& e_data) {
    this->LineParserForEverything(line, u, v, e_data);
  }

  virtual void LineParserForVFile(const std::string& line, OID_T& u,
                                  VDATA_T& u_data) {
    this->LineParserForEverything(line, u, u_data);
  }

 private:
  template <typename... Ts>
  inline const char* LineParserForEverything(const std::string& line,
                                             Ts&... vals) {
    return this->LineParserForEverything(
        line.c_str(),
        std::forward<typename std::add_lvalue_reference<Ts>::type>(vals)...);
  }

  template <typename T>
  inline const char* LineParserForEverything(const char* head, T& val) {
    return internal::match(
        head, std::forward<typename std::add_lvalue_reference<T>::type>(val));
  }

  template <typename T, typename... Ts>
  inline const char* LineParserForEverything(const char* head, T& val,
                                             Ts&... vals) {
    const char* next_head = internal::match(
        head, std::forward<typename std::add_lvalue_reference<T>::type>(val));
    return this->LineParserForEverything(
        next_head,
        std::forward<typename std::add_lvalue_reference<Ts>::type>(vals)...);
  }
};

}  // namespace grape

#endif  // GRAPE_IO_TSV_LINE_PARSER_H_
