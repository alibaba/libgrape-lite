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

#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_APP_VC_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_APP_VC_H_

#include "grape/fragment/immutable_vertexcut_fragment.h"
#include "grape/fragment/loader.h"
#include "grape/utils/memory_tracker.h"
#include "pagerank/pagerank_vc.h"
#include "utils.h"

namespace grape {

template <template <class> class APP_T, typename... Args>
void CreateAndQueryVC(const CommSpec& comm_spec, const std::string& out_prefix,
                      const ParallelEngineSpec& spec, Args... args) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  graph_spec.single_scan = FLAGS_single_scan_load;
  graph_spec.load_concurrency = FLAGS_load_concurrency;

  using FRAG_T =
      ImmutableVertexcutFragment<int64_t, grape::EmptyType, grape::EmptyType>;
  std::shared_ptr<FRAG_T> fragment = LoadVertexcutGraph<FRAG_T>(
      FLAGS_efile, FLAGS_vertex_num, comm_spec, graph_spec);
#ifdef TRACKING_MEMORY
  VLOG(1) << "[worker-" << comm_spec.worker_id() << "] after loading graph: "
          << MemoryTracker::GetInstance().GetMemoryUsageInfo();
#endif
  using AppType = APP_T<FRAG_T>;
  auto app = std::make_shared<AppType>();
  DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec, out_prefix,
                                    args...);
#ifdef TRACKING_MEMORY
  VLOG(1) << "[worker-" << comm_spec.worker_id() << "] after query: "
          << MemoryTracker::GetInstance().GetMemoryUsageInfo();
#endif
}

void RunVC() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  // FIXME: no barrier apps. more manager? or use a dynamic-cast.
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = MultiProcessSpec(comm_spec, __AFFINITY__);
  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
    if (__AFFINITY__) {
      if (spec.cpu_list.size() >= spec.thread_num) {
        spec.cpu_list.resize(spec.thread_num);
      } else {
        uint32_t num_to_append = spec.thread_num - spec.cpu_list.size();
        for (uint32_t i = 0; i < num_to_append; ++i) {
          spec.cpu_list.push_back(spec.cpu_list[i]);
        }
      }
    }
  }
  std::string name = FLAGS_application;
  if (name == "pagerank") {
    CreateAndQueryVC<PageRankVC>(comm_spec, out_prefix, spec, FLAGS_pr_d,
                                 FLAGS_pr_mr);
  } else {
    LOG(FATAL) << "No avaiable application named [" << name << "].";
  }
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_APP_VC_H_
