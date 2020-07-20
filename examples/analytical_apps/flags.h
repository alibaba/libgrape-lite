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
DECLARE_string(jobid);

DECLARE_int64(bfs_source);
DECLARE_int64(sssp_source);
DECLARE_double(pr_d);
DECLARE_int32(pr_mr);
DECLARE_int32(cdlp_mr);

DECLARE_bool(segmented_partition);
DECLARE_bool(rebalance);
DECLARE_int32(rebalance_vertex_factor);

DECLARE_bool(serialize);
DECLARE_bool(deserialize);
DECLARE_string(serialization_prefix);

DECLARE_int32(app_concurrency);

#endif  // EXAMPLES_ANALYTICAL_APPS_FLAGS_H_
