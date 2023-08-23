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

#ifndef GRAPE_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_LOADER_H_
#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <string>

#include "grape/fragment/e_fragment_loader.h"
#include "grape/fragment/ev_fragment_loader.h"
#include "grape/fragment/partitioner.h"
#include "grape/io/local_io_adaptor.h"

namespace grape {
void SetSerialize(const CommSpec& comm_spec,
                  const std::string& serialize_prefix, const std::string& efile,
                  const std::string& vfile, LoadGraphSpec& graph_spec) {
  if (serialize_prefix.empty() || efile.empty()) {
    return;
  }
  std::string digest = efile + vfile + std::to_string(comm_spec.worker_num());
  std::replace(digest.begin(), digest.end(), '/', '_');
  auto prefix =
      serialize_prefix + "/" + digest;
  char fbuf[1024];
  snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
           comm_spec.fid());

  int flag = access(fbuf, 0), sum;
  MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec.comm());

  if (sum != 0) {
    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      mkdir(prefix.c_str(), 0777);
      LOG(INFO) << "Serializing to " << prefix;
    }
    MPI_Barrier(comm_spec.comm());
    graph_spec.set_serialize(true, prefix);
  } else {
    graph_spec.set_deserialize(true, prefix);
    LOG(INFO) << "Deserializing from " << prefix;
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
          typename PARTITIONER_T = SegmentedPartitioner<typename FRAG_T::oid_t>,
          typename IOADAPTOR_T = LocalIOAdaptor,
          typename LINE_PARSER_T =
              TSVLineParser<typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
                            typename FRAG_T::edata_t>>
static std::shared_ptr<FRAG_T> LoadGraph(
    const std::string& efile, const std::string& vfile,
    const CommSpec& comm_spec,
    const LoadGraphSpec& spec = DefaultLoadGraphSpec()) {
  if (vfile.empty()) {
    std::unique_ptr<
        EFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T, LINE_PARSER_T>>
        loader(new EFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T,
                                   LINE_PARSER_T>(comm_spec));
    return loader->LoadFragment(efile, vfile, spec);
  } else {
    std::unique_ptr<
        EVFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T, LINE_PARSER_T>>
        loader(new EVFragmentLoader<FRAG_T, PARTITIONER_T, IOADAPTOR_T,
                                    LINE_PARSER_T>(comm_spec));
    return loader->LoadFragment(efile, vfile, spec);
  }
}

/**
 * 加载图变化的边数据，得到需要重算的顶点： eq: a->b, c->d, 返回： (b,
 * c)构成的vertor
 *
 */
template <typename FRAG_T, typename LINE_PARSER_T = TSVLineParser<
                               typename FRAG_T::vid_t, typename FRAG_T::vdata_t,
                               typename FRAG_T::edata_t>>
static std::unordered_set<typename FRAG_T::oid_t> LoadChageVertex(
    const std::string& change_edeg_file) {
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using line_parser_t = LINE_PARSER_T;

  line_parser_t line_parser_;
  std::unordered_set<typename FRAG_T::oid_t> changed_vertex;
  std::ifstream edegfile(change_edeg_file);
  std::string line, type;
  vid_t src, dst;

  if (!edegfile.is_open()) {
    LOG(FATAL) << "Failed to open: " << change_edeg_file;
  }
  while (getline(edegfile, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    line_parser_.LineParserForEFile(line, type, src, dst);
    if (type == "d") {
      changed_vertex.insert(dst);
    }
  }
  edegfile.close();

  return changed_vertex;
}

}  // namespace grape

#endif  // GRAPE_FRAGMENT_LOADER_H_
