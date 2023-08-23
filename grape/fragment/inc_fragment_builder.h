#ifndef AUTOINC_GRAPE_FRAGMENT_INC_FRAGMENT_BUILDER_H_
#define AUTOINC_GRAPE_FRAGMENT_INC_FRAGMENT_BUILDER_H_
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/graph/edge.h"
#include "grape/io/tsv_line_parser.h"

template <typename T>
struct std::hash<std::pair<T, T>> {
  size_t operator()(const std::pair<T, T>& pair) const noexcept {
    return pair.first ^ pair.second;
  }
};

namespace grape {

template <typename FRAG_T>
class IncFragmentBuilder {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vdata_t = typename FRAG_T::vdata_t;
  using edata_t = typename FRAG_T::edata_t;
  using edge_t = Edge<vid_t, edata_t>;

 public:
  explicit IncFragmentBuilder(std::shared_ptr<FRAG_T> fragment,
                              bool directed = true)
      : fragment_(std::move(fragment)), directed_(directed) {}

  void Init(const std::string& delta_path) {
    std::ifstream fi(delta_path);
    std::string line;
    TSVLineParser<oid_t, vdata_t, edata_t> parser;
    auto vm_ptr = fragment_->vm_ptr();
    fid_t fid = fragment_->fid();

    while (getline(fi, line)) {
      std::string type;
      oid_t u, v;
      edata_t edata;
      vid_t u_gid, v_gid;

      parser.LineParserForEFile(line, type, u, v, edata);
      CHECK(vm_ptr->GetGid(u, u_gid))
          << "Can not found src: " << u << " line: " << line;
      CHECK(vm_ptr->GetGid(v, v_gid))
          << "Can not found dst: " << v << " line: " << line;
      CHECK(type == "a" || type == "d") << "Invalid pattern: " << type;

      fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
            v_fid = vm_ptr->GetFidFromGid(v_gid);

      if (u_fid == fid || v_fid == fid) {
        if (type == "a") {
          added_edges_[u_gid].emplace(v_gid, edata);

          if (!directed_) {
            added_edges_[v_gid].emplace(u_gid, edata);
          }

        } else if (type == "d") {
          deleted_edges_[u_gid].insert(v_gid);

          if (!directed_) {
            deleted_edges_[v_gid].insert(u_gid);
          }
        }
      }
    }

    fi.close();
  }

  std::shared_ptr<FRAG_T> Build() {
    std::vector<internal::Vertex<vid_t, vdata_t>> vertices;
    std::vector<edge_t> edges;
    auto iv = fragment_->InnerVertices();
    auto vm_ptr = fragment_->vm_ptr();

    for (auto u_v_e : added_edges_) {
      auto& u = u_v_e.first;
      auto& v_edata_map = u_v_e.second;

      for (auto& v_e : v_edata_map) {
        auto v = v_e.first;
        auto& edata = v_e.second;

        edges.template emplace_back(u, v, edata);
      }
    }

    LOG(INFO) << "added_edges_: " << added_edges_.size();
    LOG(INFO) << "deleted_edges_: " << deleted_edges_.size();

    for (auto u : iv) {
      auto u_gid = fragment_->Vertex2Gid(u);
      auto u_it = deleted_edges_.find(u_gid);
      auto uve = added_edges_.find(u_gid);

      vertices.template emplace_back(u_gid, fragment_->GetData(u));

      for (auto& e : fragment_->GetOutgoingAdjList(u)) {
        auto v = e.neighbor;
        auto v_gid = fragment_->Vertex2Gid(v);
        auto edata = e.data;
        bool deleted = false;

        if (u_it != deleted_edges_.end()) {
          auto dst_set = u_it->second;

          deleted = dst_set.find(v_gid) != dst_set.end();
        }

        if (!deleted) {
          if (uve == added_edges_.end() ||
              uve->second.find(v_gid) == uve->second.end()) {
            edges.template emplace_back(u_gid, v_gid, edata);
          }
        }
      }
    }

    auto new_frag = std::make_shared<FRAG_T>(vm_ptr);

    new_frag->Init(fragment_->fid(), vertices, edges);
    return new_frag;
  }

  std::vector<std::pair<oid_t, oid_t>> GetAddedEdges() {
    std::vector<std::pair<oid_t, oid_t>> edges;
    auto vm_ptr = fragment_->vm_ptr();

    for (auto& pair : added_edges_) {
      auto u_gid = pair.first;
      oid_t u_oid;

      CHECK(vm_ptr->GetOid(u_gid, u_oid));

      for (auto& v_e : pair.second) {
        auto v_gid = v_e.first;
        oid_t v_oid;

        CHECK(vm_ptr->GetOid(v_gid, v_oid));
        edges.template emplace_back(u_oid, v_oid);
      }
    }

    return edges;
  }

  std::vector<std::pair<oid_t, oid_t>> GetDeletedEdges() {
    std::vector<std::pair<oid_t, oid_t>> edges;
    auto vm_ptr = fragment_->vm_ptr();

    for (auto& pair : deleted_edges_) {
      auto u_gid = pair.first;
      oid_t u_oid;

      CHECK(vm_ptr->GetOid(u_gid, u_oid));

      for (auto& v_gid : pair.second) {
        oid_t v_oid;

        CHECK(vm_ptr->GetOid(v_gid, v_oid));
        edges.template emplace_back(u_oid, v_oid);
      }
    }

    return edges;
  }

  std::vector<std::pair<vid_t, vid_t>> GetDeletedEdgesGid() {
    std::vector<std::pair<vid_t, vid_t>> edges;
    auto vm_ptr = fragment_->vm_ptr();

    for (auto& pair : deleted_edges_) {
      auto u_gid = pair.first;

      for (auto& v_gid : pair.second) {
        edges.template emplace_back(u_gid, v_gid);
      }
    }

    return edges;
  }

  std::vector<std::pair<vid_t, vid_t>> GetAddedEdgesGid() {
    std::vector<std::pair<vid_t, vid_t>> edges;
    auto vm_ptr = fragment_->vm_ptr();

    for (auto& pair : added_edges_) {
      auto u_gid = pair.first;

      for (auto& v_gid_edata : pair.second) {
        edges.template emplace_back(u_gid, v_gid_edata.first);
      }
    }

    return edges;
  }

 private:
  std::shared_ptr<FRAG_T> fragment_;
  bool directed_;
  std::unordered_map<vid_t, std::unordered_map<vid_t, edata_t>> added_edges_;
  std::unordered_map<vid_t, std::unordered_set<vid_t>> deleted_edges_;
};

}  // namespace grape
#endif  // AUTOINC_GRAPE_FRAGMENT_INC_FRAGMENT_BUILDER_H_
