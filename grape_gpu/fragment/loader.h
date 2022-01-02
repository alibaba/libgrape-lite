#ifndef GRAPE_GPU_FRAGMENT_LOADER_H_
#define GRAPE_GPU_FRAGMENT_LOADER_H_
#include <fcntl.h>
#include <openssl/md5.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "grape/fragment/partitioner.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/worker/comm_spec.h"
#include "grape_gpu/fragment/e_fragment_loader.h"
#include "grape_gpu/fragment/ev_fragment_loader.h"
#include "grape_gpu/fragment/g_fragment_loader.h"
#include "grape_gpu/fragment/load_graph_spec.h"

namespace grape_gpu {

// Print the MD5 sum as hex-digits.
std::string md5_sum(const std::string& file_path) {
  std::stringstream ss;
  auto fd = open(file_path.c_str(), O_RDONLY);

  if (fd >= 0) {
    struct stat statbuf {};

    if (fstat(fd, &statbuf) >= 0) {
      auto file_size = statbuf.st_size;
      auto file_buffer = static_cast<char*>(
          mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0));

      unsigned char result[MD5_DIGEST_LENGTH];

      MD5((unsigned char*) file_buffer, file_size, result);

      munmap(file_buffer, file_size);

      for (unsigned char ch : result) {
        char byte[3];
        std::sprintf(byte, "%02x", ch);
        ss << byte;
      }
    }
  }
  return ss.str();
}

std::string md5_sum_data(const std::string& data) {
  std::stringstream ss;

  unsigned char result[MD5_DIGEST_LENGTH];

  MD5((unsigned char*) data.c_str(), data.size(), result);

  for (unsigned char ch : result) {
    char byte[3];
    std::sprintf(byte, "%02x", ch);
    ss << byte;
  }
  return ss.str();
}

template <typename FRAG_T, typename PARTITIONER_T>
void SetSerialize(const grape::CommSpec& comm_spec,
                  const std::string& serialize_prefix, const std::string& efile,
                  const std::string& vfile, LoadGraphSpec& graph_spec) {
  if (serialize_prefix.empty() || efile.empty()) {
    return;
  }

  auto absolute = [](const std::string& rel_path) {
    char abs_path[PATH_MAX + 1];
    char* ptr = realpath(rel_path.c_str(), abs_path);
    CHECK(ptr != nullptr) << "Failed to access " << rel_path;
    return std::string(ptr);
  };

  std::string digest;

  if (comm_spec.worker_id() == 0) {
    auto efile_md5 = md5_sum_data(efile);
    if (efile_md5.empty()) {
      LOG(FATAL) << "Invalid efile: " << efile;
    }
    digest += efile_md5;
    if (!vfile.empty()) {
      auto vfile_md5 = md5_sum(vfile);
      if (vfile_md5.empty()) {
        LOG(FATAL) << "Invalid vfile: " << vfile;
      }
      digest += vfile_md5;
    }
    digest += std::to_string(comm_spec.fnum());
    digest += typeid(FRAG_T).name();
    digest += "directed=";
    digest += graph_spec.directed ? "true" : "false";
    digest += "rm_self_cycle=";
    digest += graph_spec.rm_self_cycle ? "true" : "false";
    if (comm_spec.fnum() > 1) {
      digest += typeid(PARTITIONER_T).name();
      digest += "rebalance=";
      digest += graph_spec.rebalance ? "true" : "false";
      digest += "rebalance_vertex_factor=";
      digest += std::to_string(graph_spec.rebalance_vertex_factor);
    }
    int digest_len = digest.size();
    MPI_Bcast(&digest_len, 1, MPI_INT, grape::kCoordinatorRank,
              comm_spec.comm());
    MPI_Bcast((void*) digest.data(), digest_len, MPI_CHAR,
              grape::kCoordinatorRank, comm_spec.comm());
  } else {
    int digest_len;
    MPI_Bcast(&digest_len, 1, MPI_INT, grape::kCoordinatorRank,
              comm_spec.comm());
    digest.resize(digest_len);
    MPI_Bcast((void*) digest.data(), digest_len, MPI_CHAR,
              grape::kCoordinatorRank, comm_spec.comm());
  }

  auto prefix = absolute(serialize_prefix) + "/" + digest;
  char fbuf[1024];
  int flag = 0, sum;

  snprintf(fbuf, sizeof(fbuf), grape::kSerializationFilenameFormat,
           prefix.c_str(), comm_spec.fid());
  flag = access(fbuf, R_OK) == 0 ? 0 : 1;

  MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());

  if (sum != 0) {
    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      mkdir(prefix.c_str(), 0777);
    }
    MPI_Barrier(comm_spec.comm());
    graph_spec.set_serialize(true, prefix);
  } else {
    graph_spec.set_deserialize(true, prefix);
  }
}

/**
 * @brief Loader manages graph loading from files.
 *
 * @tparam FRAG_T Type of Fragment
 * @tparam PARTITIONER_T, Type of partitioner, default is SegmentedPartitioner
 * @tparam IOADAPTOR_T, Type of IOAdaptor, default is LocalIOAdaptor
 * @tparam LINE_PARSER_T, Type of LineParser, default is TSVLineParser
 *
 * SegmentedPartitioner<typename FRAG_T::oid_t>
 * @param efile The input file of edges.
 * @param vfile The input file of vertices.
 * @param comm Communication world.
 * @param spec Specification to load graph.
 * @return std::shared_ptr<FRAG_T> Loadded Fragment.
 */
template <typename FRAG_T,
          typename PARTITIONER_T =
              grape::SegmentedPartitioner<typename FRAG_T::oid_t>,
          typename IOADAPTOR_T = grape::LocalIOAdaptor,
          typename LINE_PARSER_T = grape::TSVLineParser<
              typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
              typename FRAG_T::edata_t>>
static std::shared_ptr<FRAG_T> LoadGraph(
    const std::string& efile, const std::string& vfile,
    const grape::CommSpec& comm_spec,
    const std::string& serialization_prefix = "",
    LoadGraphSpec spec = DefaultLoadGraphSpec()) {
  SetSerialize<FRAG_T, PARTITIONER_T>(comm_spec, serialization_prefix, efile,
                                      vfile, spec);

  if (vfile.empty() && efile.empty()) {
    std::unique_ptr<
        GFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T, LINE_PARSER_T>>
        loader(new GFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T,
                                   LINE_PARSER_T>(comm_spec));
    return loader->LoadFragment(spec);
  } else if (vfile.empty()) {
    std::unique_ptr<
        EFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T, LINE_PARSER_T>>
        loader(new EFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T,
                                   LINE_PARSER_T>(comm_spec));
    return loader->LoadFragment(efile, spec);
  } else {
    std::unique_ptr<
        EVFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T, LINE_PARSER_T>>
        loader(new EVFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T,
                                    LINE_PARSER_T>(comm_spec));
    return loader->LoadFragment(efile, vfile, spec);
  }
}

}  // namespace grape_gpu
#endif  // GRAPE_GPU_FRAGMENT_LOADER_H_
