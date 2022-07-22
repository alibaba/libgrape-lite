
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

#ifndef EXAMPLES_ANALYTICAL_APPS_FLAGS_H_
#define EXAMPLES_ANALYTICAL_APPS_FLAGS_H_

#include <gflags/gflags_declare.h>

DECLARE_bool(directed);
DECLARE_string(application);
DECLARE_string(efile);
DECLARE_string(vfile);
DECLARE_string(out_prefix);
DECLARE_string(lb);
DECLARE_bool(mtx);
DECLARE_bool(rm_self_cycle);
DECLARE_double(wl_alloc_factor_in);
DECLARE_double(wl_alloc_factor_out_local);
DECLARE_double(wl_alloc_factor_out_remote);

DECLARE_int64(bfs_source);
DECLARE_int64(sssp_source);
DECLARE_int32(sssp_prio);
DECLARE_double(pr_d);
DECLARE_int32(pr_mr);
DECLARE_int32(cdlp_mr);

DECLARE_string(apr_async);
DECLARE_double(apr_breakdown);
DECLARE_double(apr_endure);
DECLARE_int32(apr_mc);
DECLARE_double(apr_epslion);
DECLARE_double(pr_d);
DECLARE_int32(IPC_chunk_size);
DECLARE_int32(IPC_chunk_num);
DECLARE_int32(IPC_capacity);

DECLARE_int32(scale);
DECLARE_int32(edgefactor);

DECLARE_string(partitioner);
DECLARE_bool(rebalance);
DECLARE_int32(rebalance_vertex_factor);

DECLARE_string(serialization_prefix);

DECLARE_int32(sssp_mr);
DECLARE_bool(debug);
DECLARE_int32(switch_round);
DECLARE_string(device);
DECLARE_bool(ws);
DECLARE_double(ws_k);
DECLARE_double(ws_mirror_factor);
DECLARE_double(mb_steal_factor);

DECLARE_int32(sssp_sw_round);
DECLARE_string(mg_to);
#endif  // EXAMPLES_ANALYTICAL_APPS_FLAGS_H_
