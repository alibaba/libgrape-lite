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

#ifndef GRAPE_FRAGMENT_EV_FRAGMENT_REBALANCE_LOADER_H_
#define GRAPE_FRAGMENT_EV_FRAGMENT_REBALANCE_LOADER_H_

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
class EVFragmentRebalanceLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using vertex_map_t = typename fragment_t::vertex_map_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;
  using line_parser_t = LINE_PARSER_T;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

  static_assert(std::is_base_of<LineParserBase<oid_t, vdata_t, edata_t>,
                                LINE_PARSER_T>::value,
                "LineParser type is invalid");

 public:
  explicit EVFragmentRebalanceLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec) {}

  ~EVFragmentRebalanceLoader() = default;

  std::shared_ptr<fragment_t> LoadFragment(const std::string& efile,
                                           const std::string& vfile,
                                           const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);
    if (spec.deserialize && (!spec.serialize)) {
      bool deserialized = deserializeFragment(fragment, spec);
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

    CHECK(!vfile.empty());
    {
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

    fid_t fnum = comm_spec_.fnum();
    partitioner_t partitioner(fnum, id_list);

    std::shared_ptr<vertex_map_t> vm_ptr =
        std::make_shared<vertex_map_t>(comm_spec_);
    vm_ptr->SetPartitioner(partitioner);
    vm_ptr->Init();
    auto builder = vm_ptr->GetLocalBuilder();

    for (auto id : id_list) {
      builder.add_vertex(id);
    }
    builder.finish(*vm_ptr);

    std::vector<vid_t> src_list, dst_list;
    std::vector<edata_t> edata_list;
    {
      auto io_adaptor =
          std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(efile)));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();
      std::string line;
      edata_t e_data;
      oid_t src, dst;
      vid_t src_gid, dst_gid;

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

        CHECK(vm_ptr->GetGid(src, src_gid));
        CHECK(vm_ptr->GetGid(dst, dst_gid));

        src_list.push_back(src_gid);
        dst_list.push_back(dst_gid);
        edata_list.push_back(e_data);
      }
      io_adaptor->Close();
    }

    std::vector<std::vector<int>> degree_lists(fnum);
    std::vector<std::vector<vid_t>> gid_map(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      degree_lists[i].resize(vm_ptr->GetInnerVertexSize(i), 0);
      gid_map[i].resize(vm_ptr->GetInnerVertexSize(i));
    }

    for (auto v : src_list) {
      fid_t fid = vm_ptr->GetFidFromGid(v);
      vid_t lid = vm_ptr->GetLidFromGid(v);
      ++degree_lists[fid][lid];
    }
    if (!spec.directed) {
      for (auto v : dst_list) {
        fid_t fid = vm_ptr->GetFidFromGid(v);
        vid_t lid = vm_ptr->GetLidFromGid(v);
        ++degree_lists[fid][lid];
      }
    }

    for (fid_t i = 0; i < fnum; ++i) {
      CHECK_LT(degree_lists[i].size(),
               static_cast<size_t>(std::numeric_limits<int>::max()));
      MPI_Allreduce(MPI_IN_PLACE, degree_lists[i].data(),
                    degree_lists[i].size(), MPI_INT, MPI_SUM,
                    comm_spec_.comm());
    }

    size_t total_edge_num = 0;
    size_t total_vertex_num = 0;
    for (auto& vec : degree_lists) {
      total_vertex_num += vec.size();
      for (auto d : vec) {
        total_edge_num += d;
      }
    }

    size_t total_score =
        total_edge_num + total_vertex_num * spec.rebalance_vertex_factor;
    std::vector<size_t> scores_before(fnum, 0), scores_after(fnum, 0);
    std::vector<size_t> enum_before(fnum, 0), enum_after(fnum, 0);

    fid_t mapped_fid = 0;
    vid_t mapped_lid = 0;
    size_t cur_score = 0;
    size_t expected_score = (total_score + fnum - 1) / fnum;
    vid_t cur_num = 0;
    std::vector<vid_t> vnum_list;
    for (fid_t i = 0; i < fnum; ++i) {
      vid_t vn = degree_lists[i].size();
      for (vid_t j = 0; j < vn; ++j) {
        size_t v_score = spec.rebalance_vertex_factor + degree_lists[i][j];
        cur_score += v_score;
        scores_before[i] += v_score;
        enum_before[i] += degree_lists[i][j];
        scores_after[mapped_fid] += v_score;
        enum_after[mapped_fid] += degree_lists[i][j];
        gid_map[i][j] = vm_ptr->Lid2Gid(mapped_fid, mapped_lid);
        ++cur_num;
        if (cur_score >= expected_score) {
          ++mapped_fid;
          mapped_lid = 0;
          cur_score = 0;
          vnum_list.push_back(cur_num);
          cur_num = 0;
        } else {
          ++mapped_lid;
        }
      }
    }
    if (mapped_fid == fnum) {
      CHECK_EQ(mapped_lid, 0);
    } else {
      CHECK_EQ(mapped_fid, fnum - 1);
      vnum_list.push_back(cur_num);
    }

    if (comm_spec_.worker_id() == 0) {
      LOG(INFO) << "Total score = " << total_score;
      for (fid_t i = 0; i < fnum; ++i) {
        LOG(INFO) << "[frag-" << i
                  << "]: vertex_num: " << degree_lists[i].size() << " -> "
                  << vnum_list[i] << ", edge_num: " << enum_before[i] << " -> "
                  << enum_after[i] << ", score: " << scores_before[i] << " ->"
                  << scores_after[i];
      }
    }

    for (auto& v : src_list) {
      fid_t fid = vm_ptr->GetFidFromGid(v);
      vid_t lid = vm_ptr->GetLidFromGid(v);
      v = gid_map[fid][lid];
    }
    for (auto& v : dst_list) {
      fid_t fid = vm_ptr->GetFidFromGid(v);
      vid_t lid = vm_ptr->GetLidFromGid(v);
      v = gid_map[fid][lid];
    }

    vm_ptr->UpdateToBalance(vnum_list, gid_map);

    std::vector<ShuffleOut<vid_t, vid_t, edata_t>> edges_to_frag(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      int worker_id = comm_spec_.FragToWorker(i);
      edges_to_frag[i].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag[i].SetDestination(worker_id, i);
      if (comm_spec_.worker_id() == worker_id) {
        edges_to_frag[i].DisableComm();
      }
    }

    std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices;
    std::vector<Edge<vid_t, edata_t>> processed_edges;

    std::thread edge_recv_thread([&]() {
      ShuffleIn<vid_t, vid_t, edata_t> data_in;
      data_in.Init(comm_spec_.fnum(), comm_spec_.comm(), edge_tag);
      fid_t dst_fid;
      int src_worker_id;
      while (!data_in.Finished()) {
        src_worker_id = data_in.Recv(dst_fid);
        if (src_worker_id == -1) {
          break;
        }
        CHECK_EQ(dst_fid, comm_spec_.fid());
        auto& buffers = data_in.buffers();
        foreach_rval(buffers, [&](vid_t&& src, vid_t&& dst, edata_t&& data) {
          processed_edges.emplace_back(src, dst, std::move(data));
        });
        data_in.Clear();
      }
    });

    size_t local_enum = src_list.size();
    for (size_t i = 0; i < local_enum; ++i) {
      fid_t src_fid = vm_ptr->GetFidFromGid(src_list[i]);
      fid_t dst_fid = vm_ptr->GetFidFromGid(dst_list[i]);
      edges_to_frag[src_fid].Emplace(src_list[i], dst_list[i], edata_list[i]);
      if (src_fid != dst_fid) {
        edges_to_frag[dst_fid].Emplace(src_list[i], dst_list[i], edata_list[i]);
      }
    }

    for (auto& ea : edges_to_frag) {
      ea.Flush();
    }

    edge_recv_thread.join();
    {
      auto& buffers = edges_to_frag[comm_spec_.fid()].buffers();
      foreach_rval(buffers, [&](vid_t&& src, vid_t&& dst, edata_t&& data) {
        processed_edges.emplace_back(src, dst, std::move(data));
      });
    }

    size_t vertex_num = id_list.size();
    if (!std::is_same<vdata_t, EmptyType>::value) {
      for (size_t i = 0; i < vertex_num; ++i) {
        vid_t gid;
        CHECK(vm_ptr->GetGid(id_list[i], gid));
        fid_t fid = vm_ptr->GetFidFromGid(gid);
        if (fid == comm_spec_.fid()) {
          processed_vertices.emplace_back(gid, vdata_list[i]);
        }
      }
    }

    fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr));
    fragment->Init(comm_spec_.fid(), spec.directed, processed_vertices,
                   processed_edges);

    if (!std::is_same<vdata_t, EmptyType>::value) {
      for (size_t i = 0; i < vertex_num; ++i) {
        typename fragment_t::vertex_t v;
        if (fragment->GetVertex(id_list[i], v)) {
          if (fragment->IsOuterVertex(v)) {
            fragment->SetData(v, vdata_list[i]);
          }
        }
      }
    }

    if (spec.serialize) {
      bool serialized = serializeFragment(fragment, vm_ptr, spec);
      if (!serialized) {
        VLOG(2) << "[worker-" << comm_spec_.worker_id()
                << "] Serialization failed.";
      }
    }

    return fragment;
  }

 private:
  bool existSerializationFile(const std::string& prefix) {
    char vm_fbuf[1024], frag_fbuf[1024];
    snprintf(vm_fbuf, sizeof(vm_fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);
    snprintf(frag_fbuf, sizeof(frag_fbuf), kSerializationFilenameFormat,
             prefix.c_str(), comm_spec_.fid());
    std::string vm_path = vm_fbuf;
    std::string frag_path = frag_fbuf;
    return exists_file(vm_path) && exists_file(frag_path);
  }

  bool deserializeFragment(std::shared_ptr<fragment_t>& fragment,
                           const LoadGraphSpec& spec) {
    std::string type_prefix = fragment_t::type_info();
    CHECK(spec.rebalance);
    type_prefix += ("_rb_" + std::to_string(spec.rebalance_vertex_factor));
    std::string typed_prefix = spec.deserialization_prefix + "/" + type_prefix;
    LOG(INFO) << "typed_prefix = " << typed_prefix;
    if (!existSerializationFile(typed_prefix)) {
      return false;
    }
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(typed_prefix));
    if (io_adaptor->IsExist()) {
      std::shared_ptr<vertex_map_t> vm_ptr =
          std::make_shared<vertex_map_t>(comm_spec_);
      vm_ptr->template Deserialize<IOADAPTOR_T>(typed_prefix, comm_spec_.fid());
      fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr));
      fragment->template Deserialize<IOADAPTOR_T>(typed_prefix,
                                                  comm_spec_.fid());
      return true;
    } else {
      return false;
    }
  }

  bool serializeFragment(std::shared_ptr<fragment_t> fragment,
                         std::shared_ptr<vertex_map_t> vm_ptr,
                         const LoadGraphSpec& spec) {
    std::string type_prefix = fragment_t::type_info();
    CHECK(spec.rebalance);
    type_prefix += ("_rb_" + std::to_string(spec.rebalance_vertex_factor));
    std::string typed_prefix = spec.serialization_prefix + "/" + type_prefix;
    char serial_file[1024];
    snprintf(serial_file, sizeof(serial_file), "%s/%s", typed_prefix.c_str(),
             kSerializationVertexMapFilename);
    vm_ptr->template Serialize<IOADAPTOR_T>(typed_prefix);
    fragment->template Serialize<IOADAPTOR_T>(typed_prefix);

    return true;
  }

  static constexpr int edge_tag = 6;

  CommSpec comm_spec_;
  line_parser_t line_parser_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_EV_FRAGMENT_REBALANCE_LOADER_H_
