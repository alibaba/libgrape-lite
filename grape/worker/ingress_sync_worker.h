
#ifndef GRAPE_WORKER_INGRESS_SYNC_WORKER_H_
#define GRAPE_WORKER_INGRESS_SYNC_WORKER_H_

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
#include "grape/graph/adj_list.h"
#include "grape/parallel/default_message_manager.h"
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
class IngressSyncWorker {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "IngressSyncWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = DefaultMessageManager;
  using vid_t = typename APP_T::vid_t;  //

  IngressSyncWorker(std::shared_ptr<APP_T> app,
                   std::shared_ptr<fragment_t> graph)
      : app_(app), graph_(graph) {}

  virtual ~IngressSyncWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    // verify the consistency between app and graph
    // prepare for the query
    // 建立一些发消息需要用到的索引，不必深究
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    communicator_.InitCommunicator(comm_spec.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
  }

  /**
   * 通过采样确定阈值来筛选数据
   * sample_size: 采样大小
   * return 阈值
   */
  value_t Scheduled(int sample_size) {
    auto& priority = app_->priority_;
    vid_t all_size = graph_->GetInnerVerticesNum();

    if (all_size <= sample_size) {
      return -__DBL_MAX__;
    } else {
      // 去重
      std::set<int> id_set;
      // random number generator
      std::mt19937 gen(time(0));
      std::uniform_int_distribution<> dis(
          0, all_size - 1);  // 给定范围 // 构造符合要求的随机数生成器
      // sample random pos, the sample reflect the whole data set more or less
      std::vector<int> sampled_pos;
      int i;
      // 采样
      for (i = 0; i < sample_size; i++) {
        int rand_pos = dis(gen);
        while (id_set.count(rand_pos) > 0) {
          rand_pos = dis(gen);
        }
        id_set.insert(rand_pos);
        sampled_pos.push_back(rand_pos);
      }

      // std::cout << "开始排序......." << std::endl;

      // get the cut index, everything larger than the cut will be scheduled
      sort(sampled_pos.begin(), sampled_pos.end(), compare_priority(priority));
      int cut_index = sample_size * FLAGS_portion;  // 选择阈值位置
      value_t threshold = abs(priority[Vertex<unsigned int>(
          sampled_pos[cut_index])]);  // 获得阈值, 保证一定是 >=0
      return threshold;
    }
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());  //用于一个通信子中所有进程的同步，调用函数时进程将处于等待状态，直到通信子中所有进程
                                     //都调用了该函数(MPI_Barrier)后才继续执行。

    app_->Init(*graph_);
    int step = 1;

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
    std::vector<std::pair<vertex_t, value_t>> output;
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    auto& prioritys = app_->priority_;

    value_t check_value_sum = 0;  // 记录每个检查周期value值变化量

    bool ispriority = false;  // 标记是否需要使用调度策略
    if (FLAGS_portion < 1) {
      ispriority = true;
    }

    while (true) {
      check_value_sum = 0;
      messages_.StartARound();  //

      {
        vertex_t v;
        value_t received_delta;
        // 从to_recv_中读取数据
        while (messages_.GetMessage(*graph_, v, received_delta)) {
          app_->accumulate(app_->deltas_[v],
                           received_delta);  //将delta加到对应的顶点上delta上面
        }
      }

      // 阈值
      value_t pri = -__DBL_MAX__;
      // 确定是否使用优先级
      if (ispriority) {
        // 设置采样大小
        int sample_size = 1000;
        // 获取阈值
        pri = Scheduled(sample_size);
      }

      // collect new messages
      for (auto u : inner_vertices) {  // 不加调度
        if (ispriority) {
          app_->priority(prioritys[u], values[u], deltas[u]);  // 更新priority
        }
        auto& value = values[u];
        auto& delta = deltas[u];
        if (ispriority && abs(prioritys[u]) < pri) {  // 优先级有负的
          continue;
        }
        auto oes = graph_->GetOutgoingAdjList(u);

        value_t temp = value;  // 统计本轮value差异
        app_->g_function(u, value, delta, oes,
                         output);  // 将delta发送给邻居(放入output)
        app_->accumulate(value, delta);
        check_value_sum += abs(value - temp);  // 统计本轮value差异
        delta = app_->default_v();             // clear delta

        // 将此次对fage外的顶点的更新累计后一次发送
        for (auto& e : output) {
          auto v = e.first;
          auto delta_to_send = e.second;
          app_->accumulate(deltas[v], delta_to_send);
        }
        output.clear();
      }

      // send local delta to remote
      for (auto v : outer_vertices) {
        auto& delta_to_send = deltas[v];
        //减少无用消息的发送
        if (delta_to_send == app_->default_v()) {
          continue;
        }
        messages_.SyncStateOnOuterVertex<fragment_t, value_t>(*graph_, v,
                                                              delta_to_send);
        delta_to_send = app_->default_v();
      }
      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;

      // default_work,同步一轮
      messages_.FinishARound();

      // LOG(INFO) << "进程：" << comm_spec_.worker_id() << ", check_value_sum="
      // << check_value_sum;
      if (termCheck(check_value_sum)) {
        break;
      }

      ++step;
    }
    LOG(INFO) << "同步worker结果收敛， step=" << step;
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
  bool termCheck(value_t check_value_sum) {
    // /*
    // 1.value差异
    value_t curr_delta_sum = 0;
    value_t local_delta_sum = check_value_sum;
    communicator_.Sum(local_delta_sum, curr_delta_sum);
    return curr_delta_sum <= FLAGS_termcheck_threshold;
    // */
    /*
    // 3.逐个比较点的误差
    auto inner_vertices = graph_->InnerVertices();
    double curr_wc_sum = 0;
    double local_wc_sum = 0;
    for(auto v : inner_vertices){
      local_wc_sum += abs(app_->values_[v]- app_->last_values_[v]);
      app_->last_values_[v] = app_->values_[v];
    }
    communicator_.Sum(local_wc_sum, curr_wc_sum);
    return curr_wc_sum < FLAGS_termcheck_threshold;
    */
  }

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> graph_;
  message_manager_t messages_;
  Communicator communicator_;

  CommSpec comm_spec_;

  class compare_priority {
   public:
    VertexArray<value_t, vid_t>& parent;

    compare_priority(VertexArray<value_t, vid_t>& inparent)
        : parent(inparent) {}

    bool operator()(const vid_t a, const vid_t b) {
      return abs(parent[Vertex<unsigned int>(a)]) >
             abs(parent[Vertex<unsigned int>(b)]);
    }
  };
};

}  // namespace grape

#endif  // GRAPE_WORKER_ASYNC_WORKER_H_
