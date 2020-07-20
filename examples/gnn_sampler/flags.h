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

#ifndef EXAMPLES_GNN_SAMPLER_FLAGS_H_
#define EXAMPLES_GNN_SAMPLER_FLAGS_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>

DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(out_prefix, "", "output prefix");

DEFINE_string(sampling_strategy, "random", "sampling strategy");
DEFINE_string(hop_and_num, "",
              "the hop and the numbers of neighbors to sampling.");
DEFINE_bool(enable_kafka, false, "");
DEFINE_string(input_topic, "", "kafka input topic.");
DEFINE_string(output_topic, "", "kafka output topic.");
DEFINE_string(
    broker_list, "localhost:9092",
    "list of kafka brokers, with formati: \'server1:port,server2:port,...\'.");
DEFINE_string(group_id, "grape_consumer", "kafka consumer group id.");
DEFINE_int32(partition_num, 1, "kafka topic partition number.");
DEFINE_int32(batch_size, 10000, "kafka consume messages batch size.");
DEFINE_int64(time_interval, 10, "kafka consume time interval/s");
DEFINE_bool(serialize, false, "whether to serialize loaded graph.");
DEFINE_bool(deserialize, false, "whether to deserialize graph while loading.");
DEFINE_string(serialization_prefix, "",
              "where to load/store the serialization files");

#endif  // EXAMPLES_GNN_SAMPLER_FLAGS_H_
