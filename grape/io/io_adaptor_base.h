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

#ifndef GRAPE_IO_IO_ADAPTOR_BASE_H_
#define GRAPE_IO_IO_ADAPTOR_BASE_H_

#include <string>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

/**
 * @brief IOAdaptorBase is the base class of I/O adaptors.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 */
class IOAdaptorBase {
 public:
  IOAdaptorBase() = default;

  virtual ~IOAdaptorBase() = default;

  virtual void Open() = 0;
  virtual void Open(const char* mode) = 0;
  virtual void Close() = 0;

  /**
   * @brief Configure sub-class specific items.
   * e.g.,
   * odps_access_key = abcd;
   * oss_read_concurrency = 16;
   * whether ReadLine for local location uses std::getline;
   *
   */
  virtual bool Configure(const std::string& key, const std::string& value) = 0;

  /**
   * @brief Set each worker only scan related parts of the whole file.
   *
   * for local: read with offset, from a big file.
   *
   */
  virtual bool SetPartialRead(int index, int total_parts) = 0;

  virtual bool ReadLine(std::string& line) = 0;

  virtual bool ReadArchive(OutArchive& archive) = 0;
  virtual bool WriteArchive(InArchive& archive) = 0;

  virtual bool Read(void* buffer, size_t size) = 0;
  virtual bool Write(void* buffer, size_t size) = 0;

  virtual void MakeDirectory(const std::string& path) = 0;
  virtual bool IsExist() = 0;
};
}  // namespace grape
#endif  // GRAPE_IO_IO_ADAPTOR_BASE_H_
