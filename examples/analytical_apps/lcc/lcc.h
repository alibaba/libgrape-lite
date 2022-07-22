#ifndef EXAMPLES_ANALYTICAL_APPS_LCC_LCC_H_
#define EXAMPLES_ANALYTICAL_APPS_LCC_LCC_H_
#include <iomanip>

#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class LCCContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  explicit LCCContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~LCCContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "Pagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    this->lb = app_config.lb;
    this->stage = 0;

    valid_out_degree.Init(vertices, 0);
    global_degree.Init(vertices, 0);
    filling_offset.Init(vertices, 0);
    tricnt.Init(vertices, 0);
    tricnt.H2D();

    row_offset.resize(vertices.size() + 1, 0);

    size_t n_edges = 0;
    using nbr_t = typename FRAG_T::nbr_t;
    for (auto u : frag.InnerVertices()) {
      n_edges += frag.GetLocalOutDegree(u) + frag.GetLocalInDegree(u);
    }

    messages.InitBuffer(1.5 * n_edges * sizeof(nbr_t),
                        1.5 * n_edges * sizeof(nbr_t));
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    tricnt.D2H();
    global_degree.D2H();

    for (auto v : iv) {
      double score = 0;
      if (global_degree[v] >= 2) {
        score = 2.0 * (tricnt[v]) /
                (static_cast<int64_t>(global_degree[v]) *
                 (static_cast<int64_t>(global_degree[v]) - 1));
      }
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << score << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<int, vid_t> valid_out_degree;
  VertexArray<vid_t, vid_t> global_degree;
  thrust::device_vector<int> row_offset;
  thrust::device_vector<vid_t> col_indices;
  thrust::device_vector<vid_t> col_sorted_indices;
  VertexArray<int, vid_t> filling_offset;
  VertexArray<int, vid_t> tricnt;
  int stage{};
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
};

template <typename FRAG_T>
class LCC : public GPUAppBase<FRAG_T, LCCContext<FRAG_T>>,
            public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(LCC<FRAG_T>, LCCContext<FRAG_T>, FRAG_T)
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto d_frag = frag.DeviceObject();
    auto d_global_degree = ctx.global_degree.DeviceObject();
    auto d_mm = messages.DeviceObject();
    WorkSourceRange<vertex_t> ws_in(inner_vertices.begin(),
                                    inner_vertices.size());

    ForEach(messages.stream(), ws_in, [=] __device__(vertex_t v) mutable {
      vid_t degree = d_frag.GetLocalOutDegree(v);

      d_global_degree[v] = degree;
      d_mm.template SendMsgThroughOEdges(d_frag, v, degree);
    });
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto dev_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& global_degree = ctx.global_degree;
    auto d_global_degree = global_degree.DeviceObject();
    auto d_valid_out_degree = ctx.valid_out_degree.DeviceObject();
    auto d_tricnt = ctx.tricnt.DeviceObject();
    auto d_mm = messages.DeviceObject();

    if (ctx.stage == 0) {
      ctx.stage = 1;
      // Get degree of outer vertices
      messages.template ParallelProcess<dev_fragment_t, vid_t>(
          dev_frag, [=] __device__(vertex_t v, vid_t degree) mutable {
            d_global_degree[v] = degree;
          });

      WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

      ForEachOutgoingEdge(
          stream, dev_frag, ws_in,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            vid_t u_degree = d_global_degree[u];
            vid_t v_degree = d_global_degree[v];
            vid_t u_gid = dev_frag.GetInnerVertexGid(u);
            vid_t v_gid = dev_frag.Vertex2Gid(v);

            if (u_degree > v_degree ||
                (u_degree == v_degree && u_gid < v_gid)) {
              atomicAdd(&d_valid_out_degree[u], 1);
            }
          },
          ctx.lb);

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        d_mm.template SendMsgThroughOEdges(dev_frag, v, d_valid_out_degree[v]);
      });
      messages.ForceContinue();
    } else if (ctx.stage == 1) {
      ctx.stage = 2;
      messages.template ParallelProcess<dev_fragment_t, int>(
          dev_frag, [=] __device__(vertex_t v, int degree) mutable {
            d_valid_out_degree[v] = degree;
          });

      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      // d_row_offset[0] should be 0
      int* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto size = vertices.size();

      CHECK_CUDA(cub::DeviceScan::InclusiveSum(
          d_temp_storage, temp_storage_bytes, d_valid_out_degree.data(),
          d_row_offset + 1, size, stream.cuda_stream()));
      CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      CHECK_CUDA(cub::DeviceScan::InclusiveSum(
          d_temp_storage, temp_storage_bytes, d_valid_out_degree.data(),
          d_row_offset + 1, size, stream.cuda_stream()));
      CHECK_CUDA(cudaFree(d_temp_storage));

      auto d_filling_offset = ctx.filling_offset.DeviceObject();

      CHECK_CUDA(cudaMemcpyAsync(d_filling_offset.data(), d_row_offset,
                                 sizeof(int) * size, cudaMemcpyDeviceToDevice,
                                 stream.cuda_stream()));

      auto n_filtered_edges = ctx.row_offset[size];

      LOG(INFO) << "Filtered edges: " << n_filtered_edges;

      ctx.col_indices.resize(n_filtered_edges);
      ctx.col_sorted_indices.resize(n_filtered_edges);

      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

      // filling edges with precomputed gids
      ForEachOutgoingEdge(
          stream, dev_frag, ws_in,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            vid_t u_degree = d_global_degree[u];
            vid_t v_degree = d_global_degree[v];
            vid_t u_gid = dev_frag.GetInnerVertexGid(u);
            vid_t v_gid = dev_frag.Vertex2Gid(v);

            if (u_degree > v_degree ||
                (u_degree == v_degree && u_gid < v_gid)) {
              auto pos = atomicAdd(&d_filling_offset[u], 1);
              d_col_indices[pos] = v_gid;
            }
          },
          ctx.lb);

      ForEachWithIndex(
          stream, ws_in, [=] __device__(size_t idx, vertex_t u) mutable {
            // TODO: Load balancing
            for (auto begin = d_row_offset[idx]; begin < d_row_offset[idx + 1];
                 begin++) {
              auto v_gid = d_col_indices[begin];

              d_mm.template SendMsgThroughOEdges(dev_frag, u, v_gid);
            }
          });

      stream.Sync();
      messages.ForceContinue();
    } else if (ctx.stage == 2) {
      ctx.stage = 3;
      auto d_filling_offset = ctx.filling_offset.DeviceObject();
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());

      messages.template ParallelProcess<dev_fragment_t, vid_t>(
          dev_frag, [=] __device__(vertex_t u, vid_t v_gid) mutable {
            vertex_t v;
            assert(dev_frag.IsOuterVertex(u));
            if (dev_frag.Gid2Vertex(v_gid, v)) {
              auto pos = atomicAdd(&d_filling_offset[u], 1);
              assert(pos + 1 <= d_row_offset[u.GetValue() + 1]);
              d_col_indices[pos] = v_gid;
            }
          });

      ctx.filling_offset.resize(0);

      // Sort destinations with segmented sort
      {
        WorkSourceRange<vertex_t> ws_in(vertices.begin(), vertices.size());
        int n_vertices = vertices.size();
        int n_edges = ctx.col_sorted_indices.size();
        int num_items = n_edges;
        int num_segments = n_vertices;
        auto* d_offsets = thrust::raw_pointer_cast(ctx.row_offset.data());
        auto* d_keys_in = thrust::raw_pointer_cast(ctx.col_indices.data());
        auto* d_keys_out =
            thrust::raw_pointer_cast(ctx.col_sorted_indices.data());

        auto begin = grape::GetCurrentTime();
        // Determine temporary device storage requirements
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
            num_items, num_segments, d_offsets, d_offsets + 1));
        // Allocate temporary storage
        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        // Run sorting operation
        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
            num_items, num_segments, d_offsets, d_offsets + 1));
        CHECK_CUDA(cudaFree(d_temp_storage));
        LOG(INFO) << "Sort time: " << grape::GetCurrentTime() - begin;

        // Now, we pre-compute lid
        LaunchKernel(
            stream,
            [=] __device__(vid_t * in_gid, vid_t * out_lid) {
              auto tid = TID_1D;
              auto nthreads = TOTAL_THREADS_1D;

              for (size_t eid = 0 + tid; eid < n_edges; eid += nthreads) {
                vertex_t dst;
                bool ok = dev_frag.Gid2Vertex(in_gid[eid], dst);

                assert(ok);
                out_lid[eid] = dst.GetValue();
              }
            },
            thrust::raw_pointer_cast(ctx.col_sorted_indices.data()),
            thrust::raw_pointer_cast(ctx.col_indices.data()));
      }

      {
        WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

        auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
        auto* d_col_indices =
            thrust::raw_pointer_cast(ctx.col_sorted_indices.data());
        auto* d_dst_lid = thrust::raw_pointer_cast(ctx.col_indices.data());

        // Calculate intersection
        ForEachWithIndex(
            stream, ws_in, [=] __device__(size_t idx, vertex_t u) mutable {
              int triangle_count = 0;

              for (auto eid = d_row_offset[idx]; eid < d_row_offset[idx + 1];
                   eid++) {
                vertex_t v(d_dst_lid[eid]);

                auto edge_begin_u = d_row_offset[u.GetValue()],
                     edge_end_u = d_row_offset[u.GetValue() + 1];
                auto edge_begin_v = d_row_offset[v.GetValue()],
                     edge_end_v = d_row_offset[v.GetValue() + 1];
                auto degree_u = edge_end_u - edge_begin_u;
                auto degree_v = edge_end_v - edge_begin_v;

                if (degree_u > 0 && degree_v > 0) {
                  auto min_degree = min(degree_u, degree_v);
                  auto max_degree = max(degree_u, degree_v);
                  auto total_degree = degree_u + degree_v;

                  if (min_degree * ilog2(max_degree) * 10 < total_degree) {
                    auto min_edge_begin =
                        degree_u < degree_v ? edge_begin_u : edge_begin_v;
                    auto min_edge_end = min_edge_begin + min_degree;
                    auto max_edge_begin =
                        degree_u < degree_v ? edge_begin_v : edge_begin_u;
                    ArrayView<vid_t> dst_gids(&d_col_indices[max_edge_begin],
                                              max_degree);

                    for (; min_edge_begin < min_edge_end; min_edge_begin++) {
                      auto dst_gid_from_small = d_col_indices[min_edge_begin];

                      if (BinarySearch(dst_gids, dst_gid_from_small)) {
                        // convert from dst_gid_from_small to lid without
                        // calling Gid2Vertex
                        vertex_t comm_vertex(d_dst_lid[min_edge_begin]);

                        triangle_count += 1;
                        atomicAdd(&d_tricnt[comm_vertex], 1);
                        atomicAdd(&d_tricnt[v], 1);
                      }
                    }
                  } else {
                    // traverse dsts from u and v sequentially
                    while (edge_begin_u < edge_end_u &&
                           edge_begin_v < edge_end_v) {
                      auto dst_from_u = d_col_indices[edge_begin_u];
                      auto dst_from_v = d_col_indices[edge_begin_v];

                      if (dst_from_u < dst_from_v) {
                        edge_begin_u++;
                      } else if (dst_from_u > dst_from_v) {
                        edge_begin_v++;
                      } else {
                        // convert from dst_gid_from_small to lid without
                        // calling Gid2Vertex
                        vertex_t comm_vertex(d_dst_lid[edge_begin_u]);

                        triangle_count += 1;
                        atomicAdd(&d_tricnt[comm_vertex], 1);
                        atomicAdd(&d_tricnt[v], 1);
                        edge_begin_u++;
                        edge_begin_v++;
                      }
                    }
                  }
                }
              }

              atomicAdd(&d_tricnt[u], triangle_count);
            });
      }

      {
        WorkSourceRange<vertex_t> ws_in(ov.begin(), ov.size());
        ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
          if (d_tricnt[v] != 0) {
            d_mm.template SyncStateOnOuterVertex(dev_frag, v, d_tricnt[v]);
          }
        });
      }

      messages.ForceContinue();
    } else if (ctx.stage == 3) {
      messages.template ParallelProcess<dev_fragment_t, int>(
          dev_frag, [=] __device__(vertex_t v, int tri_cnt) mutable {
            atomicAdd(&d_tricnt[v], tri_cnt);
          });
    }
  }
};
}  // namespace grape_gpu
#endif
