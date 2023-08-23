#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include "examples/analytical_apps/tornado.h"
#include "examples/analytical_apps/tornado/php.h"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./tornado_php [grape_opts]");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "analytical_apps");
    exit(1);
  }
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  grape::tornado::Init();

  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;

  timer_start(is_coordinator);

  std::string efile = FLAGS_efile;
  std::string efile_update = FLAGS_efile_update;
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

  using OID_T = int32_t;
  using VID_T = uint32_t;
  using VDATA_T = grape::EmptyType;
  using EDATA_T = float;
  using GraphType =
      grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                      grape::LoadStrategy::kOnlyOut>;
  using AppType = grape::tornado::PHP<GraphType>;

  grape::tornado::CreateAndQuery<GraphType, AppType>(
      comm_spec, efile, efile_update, vfile, out_prefix, spec, comm_spec,
      FLAGS_php_source, FLAGS_php_d, FLAGS_php_tol);

  grape::tornado::Finalize();

  google::ShutdownGoogleLogging();
}
