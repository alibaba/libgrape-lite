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

#ifndef GRAPE_IO_LOCAL_IO_ADAPTOR_H_
#define GRAPE_IO_LOCAL_IO_ADAPTOR_H_

#include <stdio.h>

#include <fstream>
#include <string>
#include <vector>

#include "grape/io/io_adaptor_base.h"

namespace grape {
class InArchive;
class OutArchive;

/**
 * @brief A default adaptor to read/write files from local locations.
 *
 */
class LocalIOAdaptor : public IOAdaptorBase {
 public:
  explicit LocalIOAdaptor(std::string location);

  ~LocalIOAdaptor() override;

  void Open() override;

  void Open(const char* mode) override;

  void Close() override;

  bool Configure(const std::string& key, const std::string& value) override;

  bool SetPartialRead(int index, int total_parts) override;

  bool ReadLine(std::string& line) override;

  bool ReadArchive(OutArchive& archive) override;

  bool WriteArchive(InArchive& archive) override;

  bool Read(void* buffer, size_t size) override;

  bool Write(void* buffer, size_t size) override;

  void MakeDirectory(const std::string& path) override;

  bool IsExist() override;

 private:
  static constexpr size_t LINE_SIZE = 65535;

  enum FileLocation {
    kFileLocationBegin = 0,
    kFileLocationCurrent = 1,
    kFileLocationEnd = 2,
  };

  int64_t tell();
  void seek(int64_t offset, FileLocation seek_from);
  bool setPartialReadImpl();

  FILE* file_;
  std::fstream fs_;
  std::string location_;
  bool using_std_getline_;
  char buff[LINE_SIZE]{};

  bool enable_partial_read_;
  std::vector<int64_t> partial_read_offset_;
  int total_parts_;
  int index_;
};
}  // namespace grape

#endif  // GRAPE_IO_LOCAL_IO_ADAPTOR_H_
