
#ifndef EXAMPLES_ANALYTICAL_APPS_TORNADO_H_
#define EXAMPLES_ANALYTICAL_APPS_TORNADO_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/inc_fragment_builder.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "examples/analytical_apps/flags.h"
#include "examples/analytical_apps/timer.h"
#include "examples/analytical_apps/tornado/pagerank.h"
#include "examples/analytical_apps/tornado/php.h"

namespace grape {
namespace tornado {

void Init() {
  if (!FLAGS_out_prefix.empty()) {
    if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
      mkdir(FLAGS_out_prefix.c_str(), 0777);
    }
  }

  if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  //  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const CommSpec& comm_spec, const std::string& efile_base,
                    const std::string& efile_update, const std::string& vfile,
                    const std::string& out_prefix,
                    const ParallelEngineSpec& spec, Args... args) {
  std::shared_ptr<FRAG_T> fragment;
  std::shared_ptr<typename APP_T::worker_t> worker;

  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  worker = APP_T::CreateWorker(app);

  int query_id = 0;

  timer_next("load graph " + efile_base);
  LOG(INFO) << "Loading fragment";
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);

  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile_base, vfile,
               graph_spec);

  fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
      efile_base, vfile, comm_spec, graph_spec);
  worker->Init(comm_spec, spec);
  worker->SetFragment(fragment);
  timer_next("run algorithm, Query: " + std::to_string(query_id++));
  worker->Query(std::forward<Args>(args)...);
  worker->Finalize();

  if (!efile_update.empty()) {
    timer_next("Reload fragment");
    IncFragmentBuilder<FRAG_T> inc_fragment_builder(fragment);

    inc_fragment_builder.Init(efile_update);
    fragment = inc_fragment_builder.Build();
    worker->Init(comm_spec, spec);
    worker->SetFragment(fragment);

    timer_next("run algorithm, Query: " + std::to_string(query_id++));
    worker->Query(std::forward<Args>(args)...);
    worker->Finalize();
    // release graph
    fragment.reset();
  }

  timer_next("Output");
  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  timer_end();
}

}  // namespace tornado
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_TORNADO_H_
