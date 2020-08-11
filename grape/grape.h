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

#ifndef GRAPE_GRAPE_H_
#define GRAPE_GRAPE_H_

#include "grape/app/auto_app_base.h"
#include "grape/app/batch_shuffle_app_base.h"
#include "grape/app/context_base.h"
#include "grape/app/parallel_app_base.h"
#include "grape/app/vertex_data_context.h"
#include "grape/parallel/auto_parallel_message_manager.h"
#include "grape/parallel/batch_shuffle_message_manager.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/parallel_message_manager.h"
#include "grape/utils/atomic_ops.h"
#include "grape/utils/vertex_array.h"
#include "grape/utils/vertex_set.h"
#include "grape/worker/auto_worker.h"
#include "grape/worker/batch_shuffle_worker.h"
#include "grape/worker/parallel_worker.h"
namespace grape {}

#endif  // GRAPE_GRAPE_H_
