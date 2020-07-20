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
DEFINE_string(jobid, "", "jobid, only used in LDBC graphanalytics.");
DEFINE_bool(directed, false, "input graph is directed or not.");

/* flags related to specific applications. */
DEFINE_int64(bfs_source, 0, "source vertex of bfs.");
DEFINE_int32(cdlp_mr, 10, "max rounds of cdlp.");
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_double(pr_d, 0.85, "damping_factor of pagerank");
DEFINE_int32(pr_mr, 10, "max rounds of pagerank");

DEFINE_bool(segmented_partition, true,
            "whether to use segmented partitioning.");
DEFINE_bool(rebalance, true, "whether to rebalance graph after loading.");
DEFINE_int32(rebalance_vertex_factor, 0, "vertex factor of rebalancing.");

DEFINE_bool(serialize, false, "whether to serialize loaded graph.");
DEFINE_bool(deserialize, false, "whether to deserialize graph while loading.");
DEFINE_string(serialization_prefix, "",
              "where to load/store the serialization files");

DEFINE_int32(app_concurrency, -1, "concurrency of application");
