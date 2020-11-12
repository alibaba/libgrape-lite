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

#include <sys/stat.h>

#include <memory>
#include <string>
#include <vector>

#include <grape/fragment/loader.h>
#include <grape/fragment/partitioner.h>
#include <grape/grape.h>

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include "append_only_edgecut_fragment.h"
#include "flags.h"
#include "kafka_consumer.h"
#include "kafka_producer.h"
#include "sampler.h"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./run_sampler [grape_opts]");
  if (argc == 1) {
    grape::gflags::ShowUsageWithFlagsRestrict(argv[0], "gnn_sampler");
    exit(1);
  }
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("gnn_sampler");
  google::InstallFailureSignalHandler();

  using oid_t = uint64_t;
  using vid_t = uint64_t;
  using vdata_t = grape::EmptyType;
  using edata_t = double;
  using graph_t =
      grape::AppendOnlyEdgecutFragment<oid_t, vid_t, vdata_t, edata_t>;
  using app_t = grape::Sampler<graph_t>;

  // init comm
  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    bool is_coordinator = (comm_spec.worker_id() == grape::kCoordinatorRank);
    auto spec = grape::DefaultParallelEngineSpec();
    spec.thread_num =
        (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
        comm_spec.local_num();

    // load graph and app
    auto graph_spec = grape::DefaultLoadGraphSpec();
    graph_spec.set_rebalance(false, 0);
    if (FLAGS_deserialize) {
      graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
    } else if (FLAGS_serialize) {
      graph_spec.set_serialize(true, FLAGS_serialization_prefix);
    }
    graph_spec.set_directed(false);
    auto fragment = grape::LoadGraph<graph_t, grape::HashPartitioner<oid_t>>(
        FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
    auto app = std::make_shared<app_t>();

    // init indices
    fragment->InitIndices();

    // pending job that commit result to the sink.
    std::unique_ptr<std::thread> job;

    std::vector<std::string> query_vertices;
    if (FLAGS_enable_kafka) {
      // dynamic graph append and make sampling
      std::unique_ptr<KafkaConsumer> consumer;
      std::shared_ptr<KafkaProducer> producer = std::make_shared<KafkaProducer>(
          FLAGS_broker_list, FLAGS_output_topic);
      KafkaOutputStream ostream(producer);
      std::vector<std::string> edge_msgs;
      if (is_coordinator) {
        consumer = std::unique_ptr<KafkaConsumer>(new KafkaConsumer(
            comm_spec.worker_id(), FLAGS_broker_list, FLAGS_group_id,
            FLAGS_input_topic, FLAGS_partition_num, FLAGS_time_interval,
            FLAGS_batch_size));
      }
      while (true) {
        query_vertices.clear();
        edge_msgs.clear();
        if (is_coordinator) {
          consumer->ConsumeMessages(query_vertices, edge_msgs);
          grape::BcastSend(query_vertices, comm_spec.comm());
        } else {
          grape::BcastRecv(query_vertices, comm_spec.comm(),
                           grape::kCoordinatorRank);
        }

        fragment->ExtendFragment(edge_msgs, comm_spec, graph_spec);
        if (!query_vertices.empty()) {
          auto worker = app_t::CreateWorker(app, fragment);
          worker->Init(comm_spec, spec);

          auto t_begin = grape::GetCurrentTime();
          worker->Query(FLAGS_sampling_strategy, FLAGS_hop_and_num,
                        query_vertices);
          LOG(INFO) << "Query: " << grape::GetCurrentTime() - t_begin << " s";

          if (job != nullptr && job->joinable()) {
            job->join();
            job.reset(nullptr);
          }
          if (job == nullptr) {
            job = std::unique_ptr<std::thread>(
                new std::thread([worker, &ostream]() {
                  worker->Output(ostream);
                  worker->Finalize();
                }));
          }
        }
      }
      ostream.close();
    } else {
      auto worker = app_t::CreateWorker(app, fragment);
      worker->Init(comm_spec, spec);
      auto t_begin = grape::GetCurrentTime();
      worker->Query(FLAGS_sampling_strategy, FLAGS_hop_and_num, query_vertices);
      LOG(INFO) << "Query: " << grape::GetCurrentTime() - t_begin << " s";
      std::ofstream ostream;
      if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
        mkdir(FLAGS_out_prefix.c_str(), 0777);
      }
      std::string output_path =
          grape::GetResultFilename(FLAGS_out_prefix, fragment->fid());
      ostream.open(output_path, std::ios::binary | std::ios::out);
      worker->Output(ostream);
      worker->Finalize();
      ostream.flush();
      ostream.close();
    }
  }
  grape::FinalizeMPIComm();
  google::ShutdownGoogleLogging();
}
