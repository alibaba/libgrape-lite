#include "examples/analytical_apps/d2ud.h"

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "examples/analytical_apps/flags.h"
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/fragment/loader.h"
#include "grape/grape.h"

void Init() {
  if (!FLAGS_out_prefix.empty()) {
    if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
      mkdir(FLAGS_out_prefix.c_str(), 0777);
    }
  }

  if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const grape::CommSpec& comm_spec, const std::string& efile,
                    const std::string& vfile, const std::string& out_prefix,
                    const grape::ParallelEngineSpec& spec, Args... args) {
  if (FLAGS_debug) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i) {
      sleep(1);
    }
  }

  LOG(INFO) << "Loading fragment";
  grape::LoadGraphSpec graph_spec = grape::DefaultLoadGraphSpec();

  graph_spec.set_directed(false);
  graph_spec.set_rebalance(false, 0);

  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      grape::LoadGraph<FRAG_T,
                       grape::SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  worker->Query(std::forward<Args>(args)...);

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
  worker->Finalize();
}

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./d2ud [grape_opts]");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "analytical_apps");
    exit(1);
  }
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  Init();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = grape::DefaultParallelEngineSpec();

  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
  } else {
    spec = MultiProcessSpec(comm_spec, false);
  }

  if (access(vfile.c_str(), 0) != 0) {
    LOG(ERROR) << "Can not access vfile, build oid set at runtime";
    vfile = "";
  }
  if (out_prefix.empty()) {
    LOG(FATAL) << "Empty out_prefix";
  }

  using OID_T = int32_t;
  using VID_T = uint32_t;
  using VDATA_T = grape::EmptyType;
  using EDATA_T = int32_t;
  using GraphType =
      grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                      grape::LoadStrategy::kOnlyOut>;
  using AppType = grape::D2UD<GraphType>;

  CreateAndQuery<GraphType, AppType, bool>(comm_spec, efile, vfile, out_prefix,
                                           spec, FLAGS_d2ud_weighted);

  google::ShutdownGoogleLogging();
}
