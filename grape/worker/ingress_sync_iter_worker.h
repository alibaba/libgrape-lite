#ifndef GRAPE_WORKER_INGRESS_SYNC_ITER_WORKER_H_
#define GRAPE_WORKER_INGRESS_SYNC_ITER_WORKER_H_

#include <grape/fragment/loader.h>

#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/fragment/inc_fragment_builder.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/parallel.h"
#include "grape/parallel/parallel_engine.h"
#include "timer.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class IterateKernel;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class IngressSyncIterWorker : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "IngressSyncIterWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using vid_t = typename APP_T::vid_t;

  IngressSyncIterWorker(std::shared_ptr<APP_T> app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());
    terminate_checking_time_ = 0;

    InitParallelEngine(pe_spec);
    LOG(INFO) << "Thread num: " << thread_num();
  }

  /**
   * 通过采样确定阈值来筛选数据
   * sample_size: 采样大小
   * return 阈值
   */
  //  value_t Scheduled(int sample_size) {
  ////    auto& priority = app_->priority_;
  //    vid_t all_size = graph_->GetInnerVerticesNum();
  //    if (all_size <= sample_size) {
  //      return -__DBL_MAX__;
  //    } else {
  //      // 去重
  //      std::unordered_set<int> id_set;
  //      // random number generator
  //      std::mt19937 gen(time(0));
  //      std::uniform_int_distribution<> dis(
  //          0, all_size - 1);  // 给定范围 // 构造符合要求的随机数生成器
  //      // sample random pos, the sample reflect the whole data set more or
  //      less std::vector<int> sampled_pos; int i;
  //      // 采样
  //      for (i = 0; i < sample_size; i++) {
  //        int rand_pos = dis(gen);
  //        while (id_set.find(rand_pos) != id_set.end()) {
  //          rand_pos = dis(gen);
  //        }
  //        id_set.insert(rand_pos);
  //        sampled_pos.push_back(rand_pos);
  //      }
  //
  //      // get the cut index, everything larger than the cut will be scheduled
  //      sort(sampled_pos.begin(), sampled_pos.end(),
  //      compare_priority(priority)); int cut_index = sample_size *
  //      FLAGS_portion;  // 选择阈值位置 value_t threshold =
  //      priority[Vertex<unsigned int>(
  //          sampled_pos[cut_index])];  // 获得阈值, 保证一定是 >=0
  //      return abs(threshold);
  //    }
  //  }

  /**
   * 用于图变换前后值的修正
   * type: -1表示回收(在旧图上)， 1表示重发(在新图上)
   *
   */
  void AmendValue(int type) {
    MPI_Barrier(comm_spec_.comm());

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    {
      messages_.StartARound();
      ForEach(inner_vertices, [this, type, &values](int tid, vertex_t u) {
        auto& value = values[u];
        auto delta = type * value;  // 发送回收值/发送补发值
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, value, delta, oes);
      });

      auto& channels = messages_.Channels();

      ForEach(outer_vertices, [this, &deltas, &channels](int tid, vertex_t v) {
        auto& delta_to_send = deltas[v];

        if (delta_to_send != app_->default_v()) {
          channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
              *graph_, v, delta_to_send);
          delta_to_send = app_->default_v();
        }
      });
      messages_.FinishARound();

      messages_.StartARound();
      messages_.template ParallelProcess<fragment_t, value_t>(
          thread_num(), *graph_,
          [this](int tid, vertex_t v, value_t received_delta) {
            app_->accumulate(app_->deltas_[v], received_delta);
          });
      // default_work,同步一轮
      messages_.FinishARound();
    }
    MPI_Barrier(comm_spec_.comm());  // 同步
  }

  /**
   * 用于重新加载图，并完成图变换前后值的校正
   *
   */
  void reloadGraph() {
    // 回收
    AmendValue(-1);

    VertexArray<value_t, vid_t> values, deltas;
    auto iv = graph_->InnerVertices();
    {
      // Backup values on old graph
      values.Init(iv);
      deltas.Init(iv);

      for (auto v : iv) {
        values[v] = app_->values_[v];
        deltas[v] = app_->deltas_[v];
      }
    }

    IncFragmentBuilder<fragment_t> inc_fragment_builder(graph_);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Parsing update file";
    }
    inc_fragment_builder.Init(FLAGS_efile_update);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Building new graph";
    }
    graph_ = inc_fragment_builder.Build();

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "New graph loaded";
    }
    app_->Init(comm_spec_, *graph_, false);
    {
      // Copy values to new graph
      for (auto v : iv) {
        app_->values_[v] = values[v];
        app_->deltas_[v] = deltas[v];
      }
    }
    app_->iterate_begin(*graph_);
    // 回收
    AmendValue(1);
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());
    app_->Init(comm_spec_, *graph_, false);

    if (FLAGS_debug) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i) {
        sleep(1);
      }
    }

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    //    auto& prioritys = app_->priority_;
    VertexArray<value_t, vid_t> last_values;

    int step = 1;
    bool batch_stage = true;

    last_values.Init(inner_vertices);

    for (auto v : inner_vertices) {
      value_t val;
      app_->init_v(v, val);
      last_values[v] = val;
    }

    double exec_time = 0;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    app_->iterate_begin(*graph_);

    while (true) {
      exec_time -= GetCurrentTime();

      messages_.StartARound();
      auto& channels = messages_.Channels();

      {
        auto begin = GetCurrentTime();
        messages_.ParallelProcess<fragment_t, value_t>(
            thread_num(), *graph_,
            [this](int tid, vertex_t v, value_t received_delta) {
              app_->accumulate(app_->deltas_[v], received_delta);
            });
        VLOG(1) << "Process time: " << GetCurrentTime() - begin;
      }

      // 阈值
      //      value_t pri = -__DBL_MAX__;
      // 确定是否使用优先级
      //      if (FLAGS_portion < 1) {
      //        // 设置采样大小
      //        int sample_size = 1000;
      //        // 获取阈值
      //        pri = Scheduled(sample_size);
      //      }

      {
        auto begin = GetCurrentTime();
        if (FLAGS_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
          parallel_for(vid_t i = inner_vertices.begin().GetValue();
                       i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            auto& value = values[u];
            auto delta = atomic_exch(deltas[u], app_->default_v());
            auto oes = graph_->GetOutgoingAdjList(u);
            app_->g_function(*graph_, u, value, delta, oes);
            app_->accumulate(value, delta);
          }
        } else {
          ForEach(inner_vertices,
                  [this, &values, &deltas](int tid, vertex_t u) {
                    // app_->priority(prioritys[u], values[u],
                    //    deltas[u]); if (abs(prioritys[u]) > pri) {
                    auto& value = values[u];
                    auto delta = atomic_exch(deltas[u], app_->default_v());
                    auto oes = graph_->GetOutgoingAdjList(u);

                    app_->g_function(*graph_, u, value, delta, oes);
                    app_->accumulate(value, delta);
                    // }
                  });
        }
        VLOG(1) << "Iter time: " << GetCurrentTime() - begin;
      }

      {
        auto begin = GetCurrentTime();
        // send local delta to remote
        ForEach(outer_vertices, [this, &deltas, &channels](int tid,
                                                           vertex_t v) {
          auto& delta_to_send = deltas[v];

          if (delta_to_send != app_->default_v()) {
            channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
                *graph_, v, delta_to_send);
            delta_to_send = app_->default_v();
          }
        });
        VLOG(1) << "Send time: " << GetCurrentTime() - begin;
      }

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;

      // default_work,同步一轮
      messages_.FinishARound();

      exec_time += GetCurrentTime();

      if (termCheck(last_values, values)) {
        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#step: " << step; 
            LOG(INFO) << "#Batch time: " << exec_time << " sec";
          }
          exec_time = 0;
          step = 1;

          if (!FLAGS_efile_update.empty()) {
            reloadGraph();  // 重新加载图
            CHECK_EQ(inner_vertices.size(), graph_->InnerVertices().size());
            inner_vertices = graph_->InnerVertices();
            outer_vertices = graph_->OuterVertices();
            CHECK_EQ(values.size(), app_->values_.size());
            CHECK_EQ(deltas.size(), app_->deltas_.size());
            values = app_->values_;
            deltas = app_->deltas_;
            //            prioritys = app_->priority_;
            continue;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
          }
          break;
        }
      }
      ++step;
    }

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        graph_->GetInnerVertex(FLAGS_sssp_source, source);
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : graph_->InnerVertices()) {
      d_sum += app_->values_[v];
      if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
        max_id = v.GetValue();
      }
    }
    LOG(INFO) << "max_d[" << graph_->GetId(vertex_t(max_id)) << "]=" << app_->values_[vertex_t(max_id)];
    LOG(INFO) << "d_sum=" << d_sum;


    MPI_Barrier(comm_spec_.comm());
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(VertexArray<value_t, vid_t>& last_values,
                 VertexArray<value_t, vid_t>& values) {
    terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum;

    for (auto u : vertices) {
      diff_sum += fabs(last_values[u] - values[u]);
      last_values[u] = values[u];
    }

    communicator_.template Sum(diff_sum, global_diff_sum);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Diff: " << global_diff_sum;
    }

    terminate_checking_time_ += GetCurrentTime();

    return global_diff_sum < FLAGS_termcheck_threshold;
  }
  bool isChange(value_t delta, vid_t c_node_num=1) {
    if (std::fabs(delta) * c_node_num 
          > FLAGS_termcheck_threshold/graph_->GetVerticesNum()) {
      return true;
    } else {
      return false;
    }
  }

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  double terminate_checking_time_;

  class compare_priority {
   public:
    VertexArray<value_t, vid_t>& parent;

    explicit compare_priority(VertexArray<value_t, vid_t>& inparent)
        : parent(inparent) {}

    bool operator()(const vid_t a, const vid_t b) {
      return abs(parent[Vertex<unsigned int>(a)]) >
             abs(parent[Vertex<unsigned int>(b)]);
    }
  };
};

}  // namespace grape

#endif  // GRAPE_WORKER_ASYNC_WORKER_H_
