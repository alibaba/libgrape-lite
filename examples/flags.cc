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

#include "flags.h"

#include <gflags/gflags.h>

/* flags related to the job. */
DEFINE_string(application, "", "application name");
DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_bool(directed, false, "input graph is directed or not.");
DEFINE_string(lb, "cm",
              "Load balancing policy, these options can be used: auto, "
              "auto_static, none, cta, cm, cmold, wm, strict");
DEFINE_bool(mtx, false,
            "Determine whether the input file is in matrix market format.");
DEFINE_bool(rm_self_cycle, false, "Remove self cycle");
DEFINE_double(wl_alloc_factor_in, 5.0,
              "Preallocate |allocation factor| * |iv num| elements for local "
              "input worklist");
DEFINE_double(wl_alloc_factor_out_local, 5.0,
              "Preallocate |allocation factor| * |iv num| elements for local "
              "output worklist");
DEFINE_double(wl_alloc_factor_out_remote, 5.0,
              "Preallocate |allocation factor| * |ov num| elements for "
              "remote output worklist");

/* flags related to specific applications. */
DEFINE_int64(bfs_source, 0, "source vertex of bfs.");
DEFINE_int32(cdlp_mr, 10, "max rounds of cdlp.");
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_int32(sssp_prio, 0, "Initial priority of sssp");
DEFINE_double(pr_d, 0.85, "damping_factor of pagerank");
DEFINE_int32(pr_mr, 10, "max rounds of pagerank");

DEFINE_double(apr_breakdown, 1e-5,
              "The threshold for async pagerank to ignore the endure");
DEFINE_double(apr_endure, 3,
              "The minimum interval between two updates in pagerank");
DEFINE_string(apr_async, "Color", "The async mode for pagerank");
DEFINE_int32(apr_mc, 4, "The max color used for async pagerank");

DEFINE_double(apr_epslion, 0.01, "");
DEFINE_string(device, "", "device list, e.g. 0,1,2,3");

DEFINE_int32(IPC_chunk_size, 128, "The chunk size used for cuda IPC");
DEFINE_int32(IPC_chunk_num, 128, "The chunk num used for cuda IPC");
DEFINE_int32(IPC_capacity, 1024, "The PCQueue capacity used for cuda IPC");

DEFINE_int32(scale, 10, "kroneckor generator: scale");
DEFINE_int32(edgefactor, 16, "kroneckor generator: edgefactor");

DEFINE_string(partitioner, "seg", "seg or hash or random");
DEFINE_bool(rebalance, true, "whether to rebalance graph after loading.");
DEFINE_int32(rebalance_vertex_factor, 0, "vertex factor of rebalancing.");

DEFINE_string(serialization_prefix, "",
              "where to load/store the serialization files");

DEFINE_bool(debug, false, "MPI Debugging");
DEFINE_bool(fuse, true, "Fused kernel");
DEFINE_int32(switch_round, 40, "");
DEFINE_bool(ws, true, "Whether to enable work-stealing");
DEFINE_double(ws_k, 20,
              "K indicates how many times a remote access is slower than the "
              "local access");
DEFINE_double(ws_mirror_factor, 0.5, "");
DEFINE_int32(sssp_mr, 30, "");
DEFINE_int32(avg_degree, 14, "");
DEFINE_double(mb_steal_factor, 0.1, "");

DEFINE_int32(sssp_sw_round, 0, "the round of SSSP switching to fused mode");
DEFINE_string(mg_to, "",
              "work migration dst devlist, e.g. migrate works from 0 to 1, and "
              "from 3 to 2, '1,-1,-1,2'");
