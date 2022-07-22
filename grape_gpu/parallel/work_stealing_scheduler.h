
#ifndef GRAPEGPU_GRAPE_GPU_PARALLEL_WORK_STEALING_SCHEDULER_H_
#define GRAPEGPU_GRAPE_GPU_PARALLEL_WORK_STEALING_SCHEDULER_H_
#include <thrust/thrust/sort.h>
#include <thrust/thrust/transform_scan.h>

#include "grape/worker/comm_spec.h"
#include "solver.h"
//#include "grape_gpu/utils/EK_solver.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/device_buffer.h"
#include "grape_gpu/utils/perf_metrics.h"

namespace grape_gpu {
template <typename FRAG_T, typename VALUE_T>
class WorkStealingScheduler {
  using vertex_t = typename FRAG_T::vertex_t;
  using value_t = VALUE_T;

 public:
  explicit WorkStealingScheduler(const FRAG_T& frag,
                                 const grape::CommSpec& comm_spec)
      : frag_(frag), comm_spec_(comm_spec), metrics_(comm_spec) {
    auto local_num = comm_spec.local_num();

    comm_spec_.Dup();
    metrics_.Evaluate();

    extra_workload_.resize(local_num, 0);
    cut_bounds_.resize(local_num);
    lengths_out_.resize(local_num);
    lengths_in_.resize(local_num * local_num);
    stolen_frontiers_.resize(local_num);
    attached_values_.resize(local_num);
    size_t tv = frag.GetVerticesNum(), max_tv;

    MPI_Allreduce(&tv, &max_tv, 1, my_MPI_SIZE_T, MPI_MAX,
                  comm_spec.local_comm());

    ps_degree_.resize(max_tv);
    attached_value_.resize(max_tv);

    for (int local_id = 0; local_id < local_num; local_id++) {
      if (local_id != comm_spec.local_id()) {
        stolen_frontiers_[local_id].resize(max_tv);
        attached_values_[local_id].resize(max_tv);
      }
    }
    attached_value_.resize(max_tv);

    ncclUniqueId id;
    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      CHECK_NCCL(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, comm_spec.comm());

    nccl_comm_ =
        std::shared_ptr<ncclComm_t>(new ncclComm_t, [](ncclComm_t* comm) {
          CHECK_NCCL(ncclCommDestroy(*comm));
          delete comm;
        });

    CHECK_NCCL(ncclCommInitRank(nccl_comm_.get(), comm_spec.local_num(), id,
                                comm_spec.local_id()));

    if (comm_spec.local_id() == 0) {
      bw_.resize(local_num);

      for (int src_id = 0; src_id < local_num; src_id++) {
        bw_[src_id].resize(local_num);
        for (int dst_id = 0; dst_id < local_num; dst_id++) {
          auto metrics = metrics_.metrics(src_id, dst_id);

          if (src_id == dst_id || !metrics.p2p || !metrics.atomic) {
            bw_[src_id][dst_id] = 0;
          } else {
            bw_[src_id][dst_id] = metrics.bi_bandwidth_gb;
          }
        }
      }

      // bw_ = {{0.0346171, 0.0572349, 0.0477931, 0.028054, 5.79673, 6.485,
      // 0.0270496, 8.10164},
      //      {0.0899805, 0.00902432, 0.0704107,
      //      0.0738412, 4.77748, 4.33356, 4.63986, 0.0239403}, {0.100076,
      //      0.0815215, 0.0553258, 0.0674163, 0.0743594, 5.90119, 0.171545,
      //      0.171645}, {0.141987, 0.113185, 0.0823987, 0.116528, 6.00978,
      //      0.0851168, 0.169577, 0.181772}, {6.66895, 4.06951, 0.13155,
      //      0.291168, 0.10935, 0.153151, 0.119405, 0.132983}, {3.752, 6.03014,
      //      0.189163, 0.0679018, 0.0501424, 0.0140574, 0.0248829, 0.0472434},
      //      {0.0240749, 0.158113, 0.304401, 3.04175, 0.0470724, 0.0405471,
      //      0.0231202, 0.0752376}, {0.179203, 0.022751, 2.98151, 0.191753,
      //      0.0236064, 0.0432746, 0.0424776, 0.0101041}};
      // bw_ = {{0.080071, 0.159303, 0.217628,
      // 0.10964, 5.79673, 6.485, 5.25637, 8.10164},
      //       {0.0639178, 0.0340539, 0.425463,
      //       0.0843792, 4.77748, 4.33356, 4.63986, 0.0365441},
      //       {2.83209, 2.94328, 0.0364054, 7.62514,
      //       0.0701377, 5.90119, 3.84762, 2.45916}, {0.629574,
      //       0.0779262, 1.46163, 0.0780942, 6.00978,
      //       0.571463, 1.37778, 2.2121}, {6.66895, 4.06951, 0.864767, 5.07799,
      //       0.0818875, 0.168069, 2.12054, 0.18754}, {3.752, 6.03014,
      //       0.189163, 0.234292, 0.049672, 0.0391134, 0.0678795, 0.0884038},
      //       {0.117185, 0.158113, 0.304401, 3.04175, 0.0630654, 0.0712455,
      //       0.0176548, 0.0752376}, {0.488815, 0.181425, 2.98151, 0.191753,
      //       0.222956, 0.0855683, 0.0946173, 0.0310146}};
      bw_ = {{0.580838, 0.377893, 0.211331, 1.70082, 5.79673, 6.485, 5.25637,
              8.10164},
             {0.195213, 0.432238, 0.730506, 0.387815, 4.77748, 4.33356, 4.63986,
              0.26787},
             {2.83209, 0.0906153, 0.452241, 7.62514, 0.563145, 5.90119, 3.84762,
              2.45916},
             {0.629574, 0.087776, 1.46163, 0.0465433, 6.00978, 0.571463,
              1.37778, 2.2121},
             {1.1967, 5.34721, 0.0898737, 1.33851, 0.0289932, 0.0774284,
              0.749606, 0.113304},
             {0.701231, 1.957, 3.08438, 0.16476, 0.083763, 0.0377196, 0.0793776,
              0.0711535},
             {3.68492, 0.234176, 0.39492, 3.04175, 0.0765214, 0.0677774,
              0.0313908, 0.0752376},
             {10.644, 0.0603501, 2.98151, 0.411692, 0.222956, 0.210441, 0.22742,
              0.0403559}};
      std::stringstream ss;
      ss << "Bandwidth matrix: " << std::endl;
      for (int src_id = 0; src_id < local_num; src_id++) {
        for (int dst_id = 0; dst_id < local_num; dst_id++) {
          ss << bw_[src_id][dst_id] << " ";
        }
        ss << std::endl;
      }
      LOG(INFO) << ss.str();
    }

    view_stolen_frontiers_.resize(local_num);
    view_attached_values_.resize(local_num);

    Stream stream;
    WarmupNccl(comm_spec, stream, nccl_comm_);

    int supportsCoopLaunch = 0;
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
                           dev);
    if (supportsCoopLaunch != 1)
      LOG(FATAL) << "Cooperative Launch is not supported on this machine "
                    "configuration.";
  }
  void ShowHint() {
    std::stringstream ss;
    ss << std::endl << "bw: " << std::endl;
    int n = bw_.size();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        ss << bw_[i][j] << " ";
      }
      ss << std::endl;
    }
    LOG(INFO) << ss.str();
  }

  const pinned_vector<thrust::pair<size_t, size_t>>& CalculateBounds(
      const Stream& stream, ArrayView<vertex_t> frontier, double k) {
    RangeMarker marker_prepare(true, "Calc workload");
    auto d_frag = frag_.DeviceObject();
    auto n_frontier = frontier.size();
    auto local_num = comm_spec_.local_num();
    auto get_degree = [=] __host__ __device__(const vertex_t& v) -> size_t {
      return d_frag.GetLocalOutDegree(v);
    };
    // ps stands for prefix sum
    ps_degree_.resize(n_frontier, false);
    auto d_ps_degree = ps_degree_.DeviceObject();
    // calculate prefix sum of out degree as workload
    thrust::transform_inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                                     frontier.begin(), frontier.end(),
                                     d_ps_degree.begin(), get_degree,
                                     thrust::plus<size_t>());

    //    thrust::sort(thrust::cuda::par.on(stream.cuda_stream()),
    //    frontier.begin(),
    //                 frontier.end(),
    //                 [=] __device__(const vertex_t& u, const vertex_t& v) {
    //                   assert(d_frag.IsInnerVertex(u));
    //                   assert(d_frag.IsInnerVertex(v));
    //                   return get_degree(u) < get_degree(v);
    //                 });
    stream.Sync();
    marker_prepare.Stop();

    RangeMarker marker_arrange(true, "Sch workload");
    size_t total_degree = 0;

    if (n_frontier > 0) {
      CHECK_CUDA(cudaMemcpyAsync(
          &total_degree, d_ps_degree.begin() + n_frontier - 1, sizeof(size_t),
          cudaMemcpyDeviceToHost, stream.cuda_stream()));
      stream.Sync();
    }

    if (comm_spec_.local_id() == 0) {
      std::vector<size_t> n_workload(local_num);

      MPI_Gather(&total_degree, 1, my_MPI_SIZE_T, n_workload.data(), 1,
                 my_MPI_SIZE_T, 0, comm_spec_.local_comm());

      // auto tr_ = EK_solver(bw_, n_workload, k);
      double solver_time = grape::GetCurrentTime();
      auto tr_ = ILP_solver(bw_, n_workload);
      // tr_ = control_solver(bw_, n_workload);
      solver_time = grape::GetCurrentTime() - solver_time;
      std::cout << "solver_time" << solver_time * 1000 << "\n";
      std::vector<size_t> one_d_tr;

      for (int local_id = 0; local_id < local_num; local_id++) {
        one_d_tr.insert(one_d_tr.end(), tr_[local_id].begin(),
                        tr_[local_id].end());
      }

      MPI_Scatter(one_d_tr.data(), local_num, my_MPI_SIZE_T,
                  extra_workload_.data(), local_num, my_MPI_SIZE_T, 0,
                  comm_spec_.local_comm());
#if 1
      std::stringstream ss;
      // ss << "workload: " << std::endl;
      // for (auto n : n_workload_) {
      //  ss << n << " ";
      //}
      ss << std::endl << "tr: " << std::endl;
      for (int i = 0; i < local_num; i++) {
        for (int j = 0; j < local_num; j++) {
          ss << tr_[i][j] << " ";
        }
        ss << std::endl;
      }
      LOG(INFO) << ss.str();
#endif
    } else {
      MPI_Gather(&total_degree, 1, my_MPI_SIZE_T, nullptr, 0, my_MPI_SIZE_T, 0,
                 comm_spec_.local_comm());
      MPI_Scatter(nullptr, 0, my_MPI_SIZE_T, extra_workload_.data(), local_num,
                  my_MPI_SIZE_T, 0, comm_spec_.local_comm());
    }
    marker_arrange.Stop();

    RangeMarker range_calc_bounds(true, "calc_bounds");
    LaunchKernel(
        stream,
        [=] __device__(int my_local_id, ArrayView<size_t> ps_degree,
                       ArrayView<size_t> extra_workload,
                       ArrayView<thrust::pair<size_t, size_t>> cut_bounds) {
          if (TID_1D == 0) {
            size_t last_begin_pos = 0;
            size_t last_begin = 0;

            for (int local_id = 0; local_id < local_num; local_id++) {
              if (local_id != my_local_id) {
                if (last_begin_pos < ps_degree.size()) {
                  auto l = last_begin_pos, r = ps_degree.size();
                  auto target = extra_workload[local_id];

                  while (l < r) {
                    auto m = l + (r - l) / 2;
                    auto m_val = ps_degree[m] - last_begin;

                    if (target <= m_val) {
                      r = m;
                    } else {
                      l = m + 1;
                    }
                  }

                  cut_bounds[local_id] = thrust::make_pair(last_begin_pos, l);
                  last_begin_pos = l;
                  if (l < ps_degree.size()) {
                    last_begin = ps_degree[l];
                  }
                } else {
                  cut_bounds[local_id] =
                      thrust::make_pair(last_begin_pos, last_begin_pos);
                }
              }
            }
            // rest work
            cut_bounds[my_local_id] =
                thrust::make_pair(last_begin_pos, ps_degree.size());
          }
        },
        comm_spec_.local_id(), ps_degree_.DeviceObject(),
        ArrayView<size_t>(extra_workload_),
        ArrayView<thrust::pair<size_t, size_t>>(cut_bounds_));
    stream.Sync();
    range_calc_bounds.Stop();
    return cut_bounds_;
  }

  void Updatebw(const std::vector<std::vector<double>>& real_times) {
    if (tr_.size() == 0) {  // init
      // for(int i=0; i<bw_.size(); ++i) {
      //  for(int j=0; j<bw_.size(); ++j) {
      //    bw_[i][j] = 1;
      //  }
      //}
      return;
    }
    int n = real_times.size();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (tr_[i][j] > 5000)
          bw_[i][j] = real_times[i][j] / tr_[i][j] * 1e8;
      }
    }
#if 0
    std::stringstream ss;
    ss << std::endl << "bw: " << std::endl;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        ss << bw_[i][j] << " ";
      }
      ss << std::endl;
    }
    LOG(INFO) << ss.str();
#endif
  }

  void ReportWork(const Stream& stream, ArrayView<vertex_t> frontier,
                  ArrayView<value_t> values, double k) {
    CalculateBounds(stream, frontier, k);
    RangeMarker marker_exch_frontier(true, "Exch Frontier");
    __exchangeFrontier__(stream, frontier, values);
    stream.Sync();
    marker_exch_frontier.Stop();
  }

  WorkSourceArray<vertex_t> GetStolenWorkSource(int local_id) {
    return WorkSourceArray<vertex_t>(view_stolen_frontiers_[local_id].data(),
                                     view_stolen_frontiers_[local_id].size());
  }

  ArrayView<value_t> GetAttachedValues(int local_id) {
    return view_attached_values_[local_id];
  }

  int local_id() const { return comm_spec_.local_id(); }

  int local_num() const { return comm_spec_.local_num(); }

  MPI_Comm local_comm() const { return comm_spec_.local_comm(); }

  void barrier() { MPI_Barrier(comm_spec_.local_comm()); }

  void __exchangeFrontier__(const Stream& stream, ArrayView<vertex_t> frontier,
                            ArrayView<value_t> values) {
    auto local_num = comm_spec_.local_num();

    for (int i = 0; i < local_num; i++) {
      auto bound = cut_bounds_[i];
      auto len = bound.second - bound.first;
      lengths_out_[i] = len;
    }
    MPI_Allgather(lengths_out_.data(), local_num, my_MPI_SIZE_T,
                  lengths_in_.data(), local_num, my_MPI_SIZE_T,
                  comm_spec_.local_comm());

    allToAll(stream, frontier, stolen_frontiers_);

    // extract the value attached with vertex
    attached_value_.resize(frontier.size(), false);
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      frontier.begin(), frontier.end(), attached_value_.data(),
                      [=] __host__ __device__(vertex_t v) -> value_t {
                        assert(v.GetValue() < values.size());
                        // local id as index
                        return values[v.GetValue()];
                      });
    allToAll(stream, attached_value_.DeviceObject(), attached_values_);

    for (int local_id = 0; local_id < local_num; local_id++) {
      if (local_id != comm_spec_.local_id()) {
        view_stolen_frontiers_[local_id] =
            stolen_frontiers_[local_id].DeviceObject();
        view_attached_values_[local_id] =
            attached_values_[local_id].DeviceObject();
      } else {
        auto bound = cut_bounds_[local_id];
        auto size = bound.second - bound.first;

        view_stolen_frontiers_[local_id] =
            ArrayView<vertex_t>(frontier.data() + bound.first, size);
        view_attached_values_[local_id] =
            ArrayView<value_t>(attached_value_.data() + bound.first, size);
      }
    }
  }

 private:
  template <typename T>
  void allToAll(const Stream& stream, ArrayView<T> in,
                std::vector<DeviceBuffer<T>>& out) {
    auto local_id = comm_spec_.local_id();
    auto local_num = comm_spec_.local_num();

    CHECK_NCCL(ncclGroupStart());
    for (int i = 1; i < local_num; i++) {
      auto src_lid = (local_id + i) % local_num;
      auto len = lengths_in_[src_lid * local_num + local_id];

      out[src_lid].resize(len, false);
      if (len > 0) {
        CHECK_NCCL(ncclRecv(out[src_lid].data(), len * sizeof(T), ncclChar,
                            src_lid, *nccl_comm_, stream.cuda_stream()));
      }
    }

    for (int i = 1; i < local_num; i++) {
      auto dst_lid = (local_id + local_num - i) % local_num;
      auto len = lengths_out_[dst_lid];
      auto bound = cut_bounds_[dst_lid];

      CHECK_EQ(bound.second - bound.first, len);
      if (len > 0) {
        CHECK_NCCL(ncclSend(in.data() + bound.first, len * sizeof(T), ncclChar,
                            dst_lid, *nccl_comm_, stream.cuda_stream()));
      }
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  grape::CommSpec comm_spec_;
  std::shared_ptr<ncclComm_t> nccl_comm_;
  PerfMetrics metrics_;
  std::vector<std::vector<double>> bw_;
  std::vector<std::vector<size_t>> tr_;
  DeviceBuffer<size_t> ps_degree_;  // prefix sum for degree

  pinned_vector<size_t> extra_workload_;
  pinned_vector<thrust::pair<size_t, size_t>> cut_bounds_;

  const FRAG_T& frag_;

  std::vector<size_t> lengths_out_;
  std::vector<size_t> lengths_in_;

  DeviceBuffer<value_t> attached_value_;
  std::vector<DeviceBuffer<vertex_t>> stolen_frontiers_;
  std::vector<DeviceBuffer<value_t>> attached_values_;

  std::vector<ArrayView<vertex_t>> view_stolen_frontiers_;
  std::vector<ArrayView<value_t>> view_attached_values_;
};
}  // namespace grape_gpu

#endif  // GRAPEGPU_GRAPE_GPU_PARALLEL_WORK_STEALING_SCHEDULER_H_
