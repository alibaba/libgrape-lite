#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "examples/analytical_apps/flags.h"
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/fragment/inc_fragment_builder.h"
#include "grape/fragment/loader.h"

void Run() {
  grape::CommSpec comm_spec;

  comm_spec.Init(MPI_COMM_WORLD);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;

  if (out_prefix.empty()) {
    LOG(FATAL) << "Empty out_prefix";
  }

  if (FLAGS_efile.empty() || FLAGS_efile_update.empty()) {
    LOG(FATAL) << "Please assign input efile/efile_update files.";
  }

  if (access(out_prefix.c_str(), 0) != 0) {
    mkdir(out_prefix.c_str(), 0777);
  }

  if (access(vfile.c_str(), 0) != 0) {
    LOG(WARNING) << "Can not access vfile, build oid set at runtime";
    vfile = "";
  }

  using OID_T = int32_t;
  using VID_T = uint32_t;
  using VDATA_T = grape::EmptyType;
  using EDATA_T = int32_t;
  using GraphType =
      grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                      grape::LoadStrategy::kOnlyOut>;

  grape::LoadGraphSpec graph_spec = grape::DefaultLoadGraphSpec();

  graph_spec.set_directed(true);
  graph_spec.set_rebalance(false, 0);

  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      grape::LoadGraph<GraphType,
                       grape::SegmentedPartitioner<typename GraphType::oid_t>>(
          efile, vfile, comm_spec, graph_spec);

  grape::IncFragmentBuilder<GraphType> inc_fragment_builder(fragment);

  LOG(INFO) << "Reading update file";
  inc_fragment_builder.Init(FLAGS_efile_update);
  LOG(INFO) << "Building new fragment";
  auto updated_frag = inc_fragment_builder.Build();

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);

  auto iv = updated_frag->InnerVertices();

  for (const auto& u : iv) {
    auto oes = updated_frag->GetOutgoingAdjList(u);

    for (auto& e : oes) {
      auto v = e.neighbor;
      auto& edata = e.data;

      ostream << updated_frag->GetId(u) << " " << updated_frag->GetId(v) << " "
              << edata << std::endl;
    }
  }

  ostream.close();
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

  grape::InitMPIComm();

  Run();

  grape::FinalizeMPIComm();
  google::ShutdownGoogleLogging();
}
