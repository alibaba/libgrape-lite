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

#ifndef GRAPE_PARALLEL_MESSAGE_MANAGER_BASE_H_
#define GRAPE_PARALLEL_MESSAGE_MANAGER_BASE_H_

#include <mpi.h>

#include "grape/config.h"

namespace grape {

struct TerminateInfo {
  void Init(fid_t fnum) {
    success = true;
    info.resize(fnum);
  }

  bool success;
  std::vector<std::string> info;
};

/**
 * @brief MessageManagerBase is the base class for message managers.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 */
class MessageManagerBase {
 public:
  MessageManagerBase() {}
  virtual ~MessageManagerBase() {}

  /**
   * @brief Initialize message manager.
   *
   * @param comm MPI_Comm object.
   */
  virtual void Init(MPI_Comm comm) = 0;

  /**
   * @brief This function will be called before Init step of applications.
   */
  virtual void Start() = 0;

  /**
   * @brief This function will be called before each evaluation step of
   * applications.
   */
  virtual void StartARound() = 0;

  /**
   * @brief This function will be called after each evaluation step of
   * applications.
   */
  virtual void FinishARound() = 0;

  /**
   * @brief This function will be called after the evaluation of applications.
   */
  virtual void Finalize() = 0;

  /**
   * @brief This function will be called by worker after a step to determine
   * whether evaluation is terminated.
   *
   * @return Whether evaluation is terminated.
   */
  virtual bool ToTerminate() = 0;

  /**
   * @brief Get size of messages sent by this message manager instance.
   * The return value is valid only after FinishARound is called.
   * StartARound will reset the value to zero.
   *
   * @return Size of messages sent by this message manager instance.
   */
  virtual size_t GetMsgSize() const = 0;

  /**
   * @brief Force continue to evaluate one more round even if all workers stop
   * sending message.
   *
   * This function can be called by applications.
   */
  virtual void ForceContinue() = 0;

  /**
   * @brief Force all workers terminate after this round of evaluation.
   *
   * This function can be called by applications.
   * @param info Termination info.
   */
  virtual void ForceTerminate(const std::string& info = "") = 0;

  /**
   * @brief This function is called to get gathered termination info after
   * evaluation finished.
   *
   * @return Termination info.
   */
  virtual const TerminateInfo& GetTerminateInfo() const = 0;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_MESSAGE_MANAGER_BASE_H_
