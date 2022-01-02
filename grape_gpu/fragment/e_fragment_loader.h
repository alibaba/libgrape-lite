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

#ifndef GRAPE_GPU_FRAGMENT_E_FRAGMENT_LOADER_H_
#define GRAPE_GPU_FRAGMENT_E_FRAGMENT_LOADER_H_

#include <mpi.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grape/fragment/partitioner.h"
#include "grape/io/line_parser_base.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/io/tsv_line_parser.h"
#include "grape/worker/comm_spec.h"
#include "grape_gpu/fragment/basic_fragment_loader.h"
#include "grape_gpu/fragment/load_graph_spec.h"

namespace grape_gpu {
/**
 * @brief EFragmentLoader is a loader to load fragments from separated
 * efile.
 *
 * @tparam FRAG_T Fragment type.
 * @tparam PARTITIONER_T Partitioner type.
 * @tparam IOADAPTOR_T IOAdaptor type.
 * @tparam LINE_PARSER_T LineParser type.
 */
template <typename FRAG_T,
          typename PARTITIONER_T =
              grape::SegmentedPartitioner<typename FRAG_T::oid_t>,
          typename IOADAPTOR_T = grape::LocalIOAdaptor,
          typename LINE_PARSER_T = grape::TSVLineParser<
              typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
              typename FRAG_T::edata_t>>
class EFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using partitioner_t = PARTITIONER_T;
  using io_adaptor_t = IOADAPTOR_T;
  using line_parser_t = LINE_PARSER_T;

  static constexpr grape::LoadStrategy load_strategy =
      fragment_t::load_strategy;

  static_assert(std::is_base_of<grape::LineParserBase<oid_t, vdata_t, edata_t>,
                                LINE_PARSER_T>::value,
                "LineParser type is invalid");

 public:
  explicit EFragmentLoader(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec), basic_fragment_loader_(comm_spec) {}

  ~EFragmentLoader() = default;

  std::shared_ptr<fragment_t> LoadFragment(const std::string& efile,
                                           const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment;
    if (spec.deserialize && (!spec.serialize)) {
      bool deserialized = basic_fragment_loader_.DeserializeFragment(
          fragment, spec.deserialization_prefix);
      int flag = 0;
      int sum = 0;
      if (!deserialized) {
        flag = 1;
      }
      MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec_.comm());
      if (sum != 0) {
        fragment.reset();
        if (comm_spec_.worker_id() == 0) {
          VLOG(2) << "Deserialization failed, start loading graph from "
                     "efile and vfile.";
        }
      } else {
        return fragment;
      }
    }

    VLOG(2) << "Constructing the oid set";
    std::unordered_set<oid_t> id_set;
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(efile)));
    io_adaptor->Open();
    std::string line;
    edata_t e_data;
    oid_t src, dst;
    size_t lineNo = 0;
    bool skip = spec.skip_first_valid_line;

    while (io_adaptor->ReadLine(line)) {
      ++lineNo;
      if (lineNo % 1000000 == 0) {
        VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                 << lineNo;
      }
      if (line.empty() || line[0] == '#' || line[0] == '%')
        continue;

      try {
        line_parser_.LineParserForEFile(line, src, dst, e_data);
      } catch (std::exception& e) {
        VLOG(1) << e.what();
        continue;
      }

      if (skip) {
        skip = false;
        VLOG(1) << "Skip line no " << lineNo << ": " << line;
        continue;
      }

      id_set.insert(src);
      id_set.insert(dst);
    }
    io_adaptor->Close();

    std::vector<oid_t> id_list(id_set.begin(), id_set.end());
    partitioner_t partitioner(comm_spec_.fnum(), id_list);

    basic_fragment_loader_.SetPartitioner(std::move(partitioner));
    basic_fragment_loader_.SetRebalance(spec.rebalance,
                                        spec.rebalance_vertex_factor);

    basic_fragment_loader_.Start();

    grape::EmptyType fake_data;

    for (auto& oid : id_set) {
      basic_fragment_loader_.AddVertex(oid, fake_data);
    }

    {
      auto io_adaptor =
          std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(efile)));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();
      std::string line;
      edata_t e_data;
      oid_t src, dst;

      size_t lineNo = 0;
      bool skip = spec.skip_first_valid_line;

      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                   << lineNo;
        }
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;

        try {
          line_parser_.LineParserForEFile(line, src, dst, e_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }

        // TODO: This is unreliable. Since all invalid line may lay in worker 0
        if (comm_spec_.worker_id() == 0 && skip) {
          skip = false;
          VLOG(1) << "Skip line no " << lineNo << ": " << line;
          continue;
        }
        if (src == dst && spec.rm_self_cycle) {
          continue;
        }
        basic_fragment_loader_.AddEdge(src, dst, e_data);

        if (!spec.directed) {
          basic_fragment_loader_.AddEdge(dst, src, e_data);
        }
      }
      io_adaptor->Close();
    }

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "] finished add vertices and edges";

    basic_fragment_loader_.ConstructFragment(fragment);

    if (spec.serialize) {
      bool serialized = basic_fragment_loader_.SerializeFragment(
          fragment, spec.serialization_prefix);
      if (!serialized) {
        VLOG(2) << "[worker-" << comm_spec_.worker_id()
                << "] Serialization failed.";
      }
    }

    return fragment;
  }

 private:
  grape::CommSpec comm_spec_;

  BasicFragmentLoader<fragment_t, partitioner_t, io_adaptor_t>
      basic_fragment_loader_;
  line_parser_t line_parser_;
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_FRAGMENT_E_FRAGMENT_LOADER_H_
