/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_APP_CONFIG_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_APP_CONFIG_H_
#ifdef __CUDACC__
#include "grape/parallel/parallel_engine.h"
namespace grape {
namespace cuda {
struct AppConfig {
  double wl_alloc_factor_in;
  double wl_alloc_factor_out_local;
  double wl_alloc_factor_out_remote;
  LoadBalancing lb;
};

template <grape::LoadStrategy LS>
struct MessageStrategyTrait {
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
};

template <>
struct MessageStrategyTrait<grape::LoadStrategy::kBothOutIn> {
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongEdgeToOuterVertex;
};

}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_APP_CONFIG_H_
