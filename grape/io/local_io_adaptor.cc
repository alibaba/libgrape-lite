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

#include "grape/io/local_io_adaptor.h"

#include <sys/stat.h>

#include <string>

#include <glog/logging.h>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

LocalIOAdaptor::LocalIOAdaptor(std::string location)
    : file_(nullptr),
      location_(std::move(location)),
      using_std_getline_(false),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {}

LocalIOAdaptor::~LocalIOAdaptor() {
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  } else if (fs_.is_open()) {
    fs_.clear();
    fs_.close();
  }
}

int64_t LocalIOAdaptor::tell() {
  if (using_std_getline_) {
    return fs_.tellg();
  } else {
    return ftell(file_);
  }
}

void LocalIOAdaptor::seek(const int64_t offset, const FileLocation seek_from) {
  if (using_std_getline_) {
    fs_.clear();
    if (seek_from == kFileLocationBegin) {
      fs_.seekg(offset, fs_.beg);
    } else if (seek_from == kFileLocationCurrent) {
      fs_.seekg(offset, fs_.cur);
    } else if (seek_from == kFileLocationEnd) {
      fs_.seekg(offset, fs_.end);
    } else {
      VLOG(1) << "invalid value, offset = " << offset
              << ", seek_from = " << seek_from;
    }
  } else {
    if (seek_from == kFileLocationBegin) {
      fseek(file_, offset, SEEK_SET);
    } else if (seek_from == kFileLocationCurrent) {
      fseek(file_, offset, SEEK_CUR);
    } else if (seek_from == kFileLocationEnd) {
      fseek(file_, offset, SEEK_END);
    } else {
      VLOG(1) << "invalid value, offset = " << offset
              << ", seek_from = " << seek_from;
    }
  }
}

void LocalIOAdaptor::Open() { return this->Open("r"); }

void LocalIOAdaptor::Open(const char* mode) {
  std::string tag = ".gz";
  size_t pos = location_.find(tag);
  if (pos != location_.size() - tag.size()) {
    if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
      int t = location_.find_last_of('/');
      if (t != -1) {
        std::string folder_path = location_.substr(0, t);
        if (access(folder_path.c_str(), 0) != 0) {
          MakeDirectory(folder_path);
        }
      }
    }
    if (using_std_getline_) {
      if (strchr(mode, 'b') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::binary | std::ios::in | std::ios::out);
      } else if (strchr(mode, 'a') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::out | std::ios::in | std::ios::app);
      } else if (strchr(mode, 'w') != NULL || strchr(mode, '+') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::out | std::ios::in | std::ios::trunc);
      } else if (strchr(mode, 'r') != NULL) {
        fs_.open(location_.c_str(), std::ios::in);
      }
    } else {
      file_ = fopen(location_.c_str(), mode);
    }
  } else {
    LOG(FATAL) << "invalid operation";
  }

  if ((using_std_getline_ && !fs_) ||
      (!using_std_getline_ && file_ == nullptr)) {
    LOG(FATAL) << "file doesn't exists. file = " << location_;
  }

  // check the partial read flag
  if (enable_partial_read_) {
    setPartialReadImpl();
  }
}

bool LocalIOAdaptor::Configure(const std::string& key,
                               const std::string& value) {
  if (key == "using_std_getline") {
    if (value == "false") {
      using_std_getline_ = false;
      return true;
    } else if (value == "true") {
      using_std_getline_ = true;
      return true;
    }
  }
  VLOG(1) << "error during configure local io adaptor with [" << key << ", "
          << value << "]";
  return false;
}

bool LocalIOAdaptor::SetPartialRead(const int index, const int total_parts) {
  // make sure that the bytes of each line of the file
  // is smaller than macro FINELINE
  if (index < 0 || total_parts <= 0 || index >= total_parts) {
    VLOG(1) << "error during set_partial_read with [" << index << ", "
            << total_parts << "]";
    return false;
  }
  if (fs_.is_open() || file_ != nullptr) {
    VLOG(2) << "WARNING!! std::set partial read after open have no effect,"
               "You probably want to set partial before open!";
    return false;
  }
  enable_partial_read_ = true;
  index_ = index;
  total_parts_ = total_parts;
  return true;
}

bool LocalIOAdaptor::setPartialReadImpl() {
  seek(0, kFileLocationEnd);
  int64_t total_file_size = tell();
  int64_t part_size = total_file_size / total_parts_;

  partial_read_offset_.resize(total_parts_ + 1, 0);
  partial_read_offset_[total_parts_] = total_file_size;

  // move breakpoint to the next of nearest character '\n'
  for (int i = 1; i < total_parts_; ++i) {
    partial_read_offset_[i] = i * part_size;

    if (partial_read_offset_[i] < partial_read_offset_[i - 1]) {
      partial_read_offset_[i] = partial_read_offset_[i - 1];
    } else {
      // traversing backwards to find the nearest character '\n',
      seek(partial_read_offset_[i], kFileLocationBegin);
      int dis = 0;
      while (true) {
        char buffer[1];
        std::memset(buff, 0, sizeof(buffer));
        bool status = Read(buffer, 1);
        if (!status || buffer[0] == '\n') {
          break;
        } else {
          dis++;
        }
      }
      // move to next character of '\n'
      partial_read_offset_[i] += (dis + 1);
      if (partial_read_offset_[i] > total_file_size) {
        partial_read_offset_[i] = total_file_size;
      }
    }
  }

  int64_t file_stream_pos = partial_read_offset_[index_];
  seek(file_stream_pos, kFileLocationBegin);
  return true;
}

bool LocalIOAdaptor::ReadLine(std::string& line) {
  if (enable_partial_read_ && tell() >= partial_read_offset_[index_ + 1]) {
    return false;
  }
  if (using_std_getline_) {
    getline(fs_, line);
    return !line.empty();
  } else {
    if (file_ && fgets(buff, LINE_SIZE, file_)) {
      std::string str(buff);
      line.swap(str);
      return true;
    } else {
      return false;
    }
  }
}

bool LocalIOAdaptor::ReadArchive(OutArchive& archive) {
  if (!using_std_getline_ && file_) {
    size_t length;
    bool status = fread(&length, sizeof(size_t), 1, file_);
    if (!status) {
      return false;
    }
    archive.Allocate(length);
    status = fread(archive.GetBuffer(), 1, length, file_);
    return status;
  } else {
    VLOG(1) << "invalid operation.";
    return false;
  }
}

bool LocalIOAdaptor::WriteArchive(InArchive& archive) {
  if (!using_std_getline_ && file_) {
    size_t length = archive.GetSize();
    bool status = fwrite(&length, sizeof(size_t), 1, file_);
    if (!status) {
      return false;
    }
    status = fwrite(archive.GetBuffer(), 1, length, file_);
    if (!status) {
      return false;
    }
    fflush(file_);
    return true;
  } else {
    VLOG(1) << "invalid operation.";
    return false;
  }
}

bool LocalIOAdaptor::Read(void* buffer, size_t size) {
  if (using_std_getline_) {
    fs_.read(static_cast<char*>(buffer), size);
    if (!fs_) {
      return false;
    }
  } else {
    if (file_) {
      bool status = fread(buffer, 1, size, file_);
      if (!status) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool LocalIOAdaptor::Write(void* buffer, size_t size) {
  if (using_std_getline_) {
    fs_.write(static_cast<char*>(buffer), size);
    if (!fs_) {
      return false;
    }
    fs_.flush();
  } else {
    if (file_) {
      bool status = fwrite(buffer, 1, size, file_);
      if (!status) {
        return false;
      }
      fflush(file_);
    } else {
      return false;
    }
  }
  return true;
}

void LocalIOAdaptor::Close() {
  if (using_std_getline_) {
    if (fs_.is_open()) {
      fs_.close();
    }
  } else {
    if (file_ != nullptr) {
      fclose(file_);
      file_ = nullptr;
    }
  }
}

void LocalIOAdaptor::MakeDirectory(const std::string& path) {
  std::string dir = path;
  int len = dir.size();
  if (dir[len - 1] != '/') {
    dir[len] = '/';
    len++;
  }
  std::string temp;
  for (int i = 1; i < len; i++) {
    if (dir[i] == '/') {
      temp = dir.substr(0, i);
      if (access(temp.c_str(), 0) != 0) {
        if (mkdir(temp.c_str(), 0777) != 0) {
          VLOG(1) << "failed operaiton.";
        }
      }
    }
  }
}

bool LocalIOAdaptor::IsExist() { return access(location_.c_str(), 0) == 0; }

}  // namespace grape
