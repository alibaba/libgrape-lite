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

#include <memory>
#include <string>

#include "grape/fragment/e_fragment_loader.h"
#include "grape/fragment/ev_fragment_loader.h"
#include "grape/fragment/partitioner.h"
#include "grape/io/local_io_adaptor.h"

namespace grape {
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

}  // namespace grape

#endif  // GRAPE_FRAGMENT_LOADER_H_
