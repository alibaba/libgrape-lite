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

#include "mutate_benchmark_flags.h"

#include <gflags/gflags.h>

DEFINE_string(vfile, "", "vertex file");
DEFINE_string(efile, "", "edge file");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_string(delta_efile_prefix, "", "delta edge file directory");
DEFINE_int32(delta_efile_part_num, 0, "delta edge file part num");
DEFINE_int64(sssp_source, 0, "source vertex of sssp");
DEFINE_bool(directed, false, "input graph is directed or not.");
DEFINE_double(pr_d, 0.85, "damping_factor of pagerank");
DEFINE_int32(pr_mr, 10, "max rounds of pagerank");
DEFINE_string(application, "", "application name");
