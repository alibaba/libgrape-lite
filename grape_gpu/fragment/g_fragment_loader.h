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

#ifndef GRAPE_GPU_FRAGMENT_G_FRAGMENT_LOADER_H_
#define GRAPE_GPU_FRAGMENT_G_FRAGMENT_LOADER_H_

#include <mpi.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "grape/fragment/partitioner.h"
#include "grape/io/line_parser_base.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/io/tsv_line_parser.h"
#include "grape/worker/comm_spec.h"
#include "grape_gpu/fragment/basic_fragment_loader.h"
#include "grape_gpu/fragment/load_graph_spec.h"
#include "grape_gpu/utils/kronecker.h"

namespace grape_gpu {
/**
 * @brief GFragmentLoader is a loader to load fragments from kronecker
 * generator.
 *
 * @tparam FRAG_T Fragment type.
 * @tparam PARTITIONER_T Partitioner type.
 * @tparam IOADAPTOR_T IOAdaptor type.
 * @tparam LINE_PARSER_T LineParser type.
 */

template<typename FRAG_T, typename Enable = void>
struct KronecEdge{};

template<typename FRAG_T>
struct KronecEdge<FRAG_T, typename std::enable_if<
            !std::is_same<typename FRAG_T::edata_t, grape::EmptyType>::value>::type>{
  using edata_t = typename FRAG_T::edata_t;
  void rand(edata_t& ed){
    auto rd = [=](){
      srand(time(NULL));
      return std::rand()%63;
    };
    ed = rd();
  }
};

template<typename FRAG_T>
struct KronecEdge<FRAG_T, typename std::enable_if<
            std::is_same<typename FRAG_T::edata_t, grape::EmptyType>::value>::type>{
  using edata_t = typename FRAG_T::edata_t;
  void rand(edata_t& ed){}
};


template <typename FRAG_T,
          typename PARTITIONER_T =
              grape::SegmentedPartitioner<typename FRAG_T::oid_t>,
          typename IOADAPTOR_T = grape::LocalIOAdaptor,
          typename LINE_PARSER_T = grape::TSVLineParser<
              typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
              typename FRAG_T::edata_t>>
class GFragmentLoader {
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
  explicit GFragmentLoader(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec), basic_fragment_loader_(comm_spec) {}

  ~GFragmentLoader() = default;

  std::shared_ptr<fragment_t> LoadFragment(const LoadGraphSpec& spec) {
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

    //std::unordered_set<oid_t> id_set;
    //std::vector<oid_t> id_list(id_set.begin(), id_set.end());
    std::vector<oid_t> id_list;
    partitioner_t partitioner(comm_spec_.fnum(), id_list);

    basic_fragment_loader_.SetPartitioner(std::move(partitioner));
    basic_fragment_loader_.SetRebalance(spec.rebalance,
                                        spec.rebalance_vertex_factor);

    basic_fragment_loader_.Start();

    grape::EmptyType fake_data;

    {
      uint32_t* raw_distributed_edgelist = nullptr;
      int64_t* raw_size = (int64_t*)malloc(sizeof(int64_t));
      kronecker(spec.kronec_scale, spec.kronec_edgefactor, raw_size, &raw_distributed_edgelist);
      int64_t T = *raw_size;
      std::string line;
      edata_t e_data;
      oid_t src, dst;
      size_t lineNo = 0;

      
      for(uint64_t idx = 0; idx < T; idx++) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][kronec] "
                   << lineNo;
        }
        try {
          src = raw_distributed_edgelist[idx<<1];
          dst = raw_distributed_edgelist[idx<<1|1];
          KronecEdge<FRAG_T> ke;
          ke.rand(e_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }

        basic_fragment_loader_.AddEdge(src, dst, e_data);
        basic_fragment_loader_.AddVertex(src, fake_data);
        basic_fragment_loader_.AddVertex(dst, fake_data);

        if (!spec.directed) {
          basic_fragment_loader_.AddEdge(dst, src, e_data);
        }
      }
      free(raw_size);
      if(raw_distributed_edgelist != nullptr) free(raw_distributed_edgelist);
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

#endif  // GRAPE_GPU_FRAGMENT_G_FRAGMENT_LOADER_H_
