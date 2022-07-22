#include <gflags/gflags.h>
#include <glog/logging.h>

#include "examples/pc_queue.h"
#include "examples/IPC.h"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  testing::InitGoogleTest(&argc, argv);
  gflags::SetUsageMessage("Usage: mpiexec [mpi_opts] ./test [grape_opts]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("analytical_apps");
  google::InstallFailureSignalHandler();
  int res = RUN_ALL_TESTS();

  google::ShutdownGoogleLogging();
  return res;
}
