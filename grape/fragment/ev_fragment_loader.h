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

#ifndef GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_

#include <mpi.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grape/fragment/basic_fragment_loader.h"
#include "grape/fragment/partitioner.h"
#include "grape/io/line_parser_base.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/io/tsv_line_parser.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief EVFragmentLoader is a loader to load fragments from separated
 * efile and vfile.
 *
 * @tparam FRAG_T Fragment type.
 * @tparam IOADAPTOR_T IOAdaptor type.
 * @tparam LINE_PARSER_T LineParser type.
 */
template <typename FRAG_T, typename IOADAPTOR_T = LocalIOAdaptor,
          typename LINE_PARSER_T =
              TSVLineParser<typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
                            typename FRAG_T::edata_t>>
class EVFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using vertex_map_t = typename fragment_t::vertex_map_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;
  using io_adaptor_t = IOADAPTOR_T;
  using line_parser_t = LINE_PARSER_T;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

  static_assert(std::is_base_of<LineParserBase<oid_t, vdata_t, edata_t>,
                                LINE_PARSER_T>::value,
                "LineParser type is invalid");

 public:
  explicit EVFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec), basic_fragment_loader_(comm_spec) {}

  ~EVFragmentLoader() = default;

  std::shared_ptr<fragment_t> LoadFragment(const std::string& efile,
                                           const std::string& vfile,
                                           const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);
    CHECK(!spec.rebalance);
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

    std::vector<oid_t> id_list;
    std::vector<vdata_t> vdata_list;
    if (!vfile.empty()) {
      auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(vfile));
      io_adaptor->Open();
      std::string line;
      vdata_t v_data;
      oid_t vertex_id;
      size_t line_no = 0;
      while (io_adaptor->ReadLine(line)) {
        ++line_no;
        if (line_no % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][vfile] "
                   << line_no;
        }
        if (line.empty() || line[0] == '#')
          continue;
        try {
          line_parser_.LineParserForVFile(line, vertex_id, v_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }
        id_list.push_back(vertex_id);
        vdata_list.push_back(v_data);
      }
      io_adaptor->Close();
    }

    partitioner_t partitioner(comm_spec_.fnum(), id_list);

    basic_fragment_loader_.SetPartitioner(std::move(partitioner));

    basic_fragment_loader_.Start();

    {
      size_t vnum = id_list.size();
      for (size_t i = 0; i < vnum; ++i) {
        basic_fragment_loader_.AddVertex(id_list[i], vdata_list[i]);
      }
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
      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                   << lineNo;
        }
        if (line.empty() || line[0] == '#')
          continue;

        try {
          line_parser_.LineParserForEFile(line, src, dst, e_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }

        basic_fragment_loader_.AddEdge(src, dst, e_data);
      }
      io_adaptor->Close();
    }

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "] finished add vertices and edges";

    basic_fragment_loader_.ConstructFragment(fragment, spec.directed);

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
  CommSpec comm_spec_;

  BasicFragmentLoader<fragment_t, io_adaptor_t> basic_fragment_loader_;
  line_parser_t line_parser_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_
