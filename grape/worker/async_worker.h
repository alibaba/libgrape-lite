
#ifndef GRAPE_WORKER_ASYNC_WORKER_H_
#define GRAPE_WORKER_ASYNC_WORKER_H_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/async_message_manager.h"
#include "grape/parallel/parallel_engine.h"

//
// #include<bits/stdc++.h>
#include <random>
#include <iostream>
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
class AsyncWorker {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "AsyncWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = AsyncMessageManager;
  using vid_t = typename APP_T::vid_t; // 

  AsyncWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app), graph_(graph) {}

  virtual ~AsyncWorker() = default;

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
    last_delta_sum = 0;
  }


/**
 * 通过采样确定阈值来筛选数据
 * sample_size: 采样大小
 * return 阈值
*/
  value_t Scheduled(int sample_size){
    // 获取全部内部点
    // auto inner_vertices = graph_->InnerVertices();
    // auto& values = app_->values_;

    auto& priority = app_->priority_;
    auto defaultv = app_->default_v();
    auto minDelta = app_->min_delta();
    // 
    vid_t all_size = graph_->GetInnerVerticesNum();
    // bool bfilter = false;

    if(all_size <= sample_size){
      return -__DBL_MAX__;
    }
    else{
      // 去重
      std::set<int> id_set;
      //random number generator
      std::mt19937 gen(time(0));
      std::uniform_int_distribution<> dis(0, all_size-1); // 给定范围 // 构造符合要求的随机数生成器 
      //sample random pos, the sample reflect the whole data set more or less
      std::vector<int> sampled_pos;
      int i;
      // 采样
      for(i=0; i<sample_size; i++){
          int rand_pos = dis(gen);
          while(id_set.count(rand_pos) > 0){
            rand_pos = dis(gen);
          }
          id_set.insert(rand_pos);
          sampled_pos.push_back(rand_pos);
      }

      // std::cout << "开始排序......." << std::endl;

      // get the cut index, everything larger than the cut will be scheduled
      sort(sampled_pos.begin(), sampled_pos.end(), compare_priority(priority));
      int cut_index = sample_size*FLAGS_portion;  // 选择阈值位置
      value_t threshold = priority[Vertex<unsigned int>(sampled_pos[cut_index])]; // 获得阈值
      return threshold; 
    }
    //  std::cout << "调度完成......." << std::endl;
  }


  void Query() {
    MPI_Barrier(comm_spec_.comm()); //用于一个通信子中所有进程的同步，调用函数时进程将处于等待状态，直到通信子中所有进程 都调用了该函数(MPI_Barrier)后才继续执行。
    // 单独的线程去进行数据的实际接收和发送
    send_th_ = std::thread([this]() { messages_.Start(); }); 
    receive_th_ = std::thread([this]() { messages_.ReceiveThread(); });
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

    bool running = true; // 标记是否运行
    // 总的内部点的delta_pr和：
    value_t delta_sum = 0;
    // 统计发送和接收消息的数量
    int send_num = 0;
    int rece_num = 0;
    value_t check_delta_sum = 0;  // 记录每个检查周期接收的总和
    
    int get_null_num = 0; // 统计每个work在循环中获取消息会零的次数
    double waite_check_time = 0;
    // 初始化接收check消息
    for(int i = 0; i < comm_spec_.worker_num(); i++){
        check_sum_values.push_back(0x3fffffff);
    }
    last_termcheck_ = timer();  // 初始化时间
    while (running) {
      bool isget = false;
      {
        vertex_t v;
        value_t received_delta;
        // 从to_recv_中读取数据
        while (messages_.GetMessage(*graph_, v, received_delta)) {
          app_->accumulate(app_->deltas_[v], received_delta);  //将delta加到对应的顶点上delta上面
          rece_num++;
          isget = true;
        }
        if(!isget){
          get_null_num++;
          messages_.Sleep(no_message_waite_time);  // 当前没有接收到消息，等待时间
        }
      }
      // 测试
      delta_sum -= check_delta_sum;
      // 阈值
      value_t pri = -__DBL_MAX__;
      // 确定是否使用优先级
      if (FLAGS_portion < 1){
        // 设置采样大小
        int sample_size = 1000;
        // 获取阈值
        pri = Scheduled(sample_size);
        // LOG(INFO) << "阈值： " << pri;
      }

      // collect new messages
      for (auto u : inner_vertices) {    // 不加调度
        app_->priority(prioritys[u], values[u], deltas[u]);  // 更新priority
        auto& value = values[u];
        auto& delta = deltas[u];
        if(abs(prioritys[u]) < abs(pri)){  // 优先级有负的
          continue;
        }
        auto oes = graph_->GetOutgoingAdjList(u);

        check_delta_sum += delta;  // 累计发送的和
        app_->g_function(u, value, delta, oes, output); // 将delta发送给邻居(放入output) 
        app_->accumulate(value, delta);
        delta = app_->default_v();  // clear delta

        // 将此次对fage外的顶点的更新累计后一次发送
        for (auto& e : output) {
          auto v = e.first;
          auto delta_to_send = e.second;

          app_->accumulate(deltas[v], delta_to_send);
        }

        output.clear();
      }
      // 测试
      delta_sum += check_delta_sum;
      // send local delta to remote
      for (auto v : outer_vertices) {
        auto& delta_to_send = deltas[v];
        //减少无用消息的发送
        if(deltas[v] <= 0){
          continue;
        }
        send_num++;
        messages_.SyncStateOnOuterVertex<fragment_t, value_t>(*graph_, v,
                                                              delta_to_send);      // 可以修改为直接发送: 
        delta_to_send = app_->default_v();
      }
      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      ++step;

      // 定期自动检测
      if(running && timer() - last_termcheck_ > termcheck_interval){
        last_termcheck_ = timer();
        waite_check_time -= last_termcheck_;
//        bool check = termCheck(check_delta_sum);
        waite_check_time += timer();
        // if(check){  // 阻塞检测时使用
        //   running = false;
        //   break;
        // }
        check_delta_sum = 0; // 每次检查完清空
      }
      if(messages_.Get_is_get_Stop()){ 
        int local = 1;
        int curr = 0;
        printf("%d 等待结束！\n", comm_spec_.worker_id());
        waite_check_time -= timer();
        communicator_.Sum(local, curr); // 同步结束，保证都接收了stop消息
        waite_check_time += timer();
        LOG(INFO) << "进程：" << comm_spec_.worker_id() << ", curr=" << curr;
        if(curr == comm_spec_.worker_num()){
          break;
        }
      }
      // if(step > 1000){
      //   break;
      // }
    }
    LOG(INFO) << comm_spec_.worker_id() << "结果收敛， step=" << step << ", get_null_num："<< get_null_num << ", 检查时间：" << waite_check_time;
    messages_.Stop();
    send_th_.join();    // 发送接收线程
    receive_th_.join(); 
    MPI_Barrier(comm_spec_.comm());
    LOG(INFO) << "进程：" << comm_spec_.worker_id() << ", delta_sum=" << delta_sum << ", send_num=" << send_num << ", rece_num=" << rece_num;
    // LOG(INFO) << "我是线程：" << comm_spec_.worker_id()  << ", work_num: " << comm_spec_.worker_num();
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
  bool termCheck(value_t check_delta_sum) {
    // /*
    //1.非阻塞发送检查信息
    value_t curr_delta_sum = 0;
    if(0 == comm_spec_.worker_id()){ // master
      // 更新的结果，统计结果，决定是否发出迭代终止
      curr_delta_sum = check_delta_sum;
      // 更新
      fid_t v;
      value_t received_check_value;
      while (messages_.GetMessage(v, received_check_value)) {
        check_sum_values[v] = received_check_value;
      }
      // 统计
      for(int i = 1; i < comm_spec_.worker_num(); i++){
        curr_delta_sum += check_sum_values[i];
        // printf("获取的值：%d %f\n", i, check_sum_values[i]);
      }
      if(curr_delta_sum < FLAGS_termcheck_threshold){ // 达到终止条件，全部终止
        for(int i = 1; i < comm_spec_.worker_num(); i++){
          messages_.SendToFragment<int>(i, 0, 1); // 1: stop
          // printf("%d发送停止消息\n", i);
          // messages_.SendToFragment_block<int>(i, 0, 1); // 1: stop
          printf("%d发送完毕！\n", i);
        }
        messages_.Set_is_get_Stop(); // 0号线程，标记停止
        // messages_.Stop2();
        printf("%d发出停止信号！！！\n", messages_.Get_is_get_Stop());
        return true;
      }
    }
    else{  // worker
      // 发出当前结果
      messages_.SendToFragment<value_t>(0, check_delta_sum, 4); // 4:提交check消息
    }
    return false;
    // return curr_delta_sum < FLAGS_termcheck_threshold;
    // */
    /*
    //2.
    double curr_delta_sum = 0;
    double local_delta_sum = check_delta_sum;
    communicator_.Sum(local_delta_sum, curr_delta_sum); // 阻塞求和
    auto diff = abs(curr_delta_sum - last_delta_sum);
    // LOG(INFO) << "terminate check : last progress " << last_delta_sum
    //           << " current progress " << curr_delta_sum << " difference "
    //           << diff;
    // last_delta_sum = curr_delta_sum;
    return curr_delta_sum < FLAGS_termcheck_threshold;
    */
    /*
    // 3.逐个比较点的误差
    auto inner_vertices = graph_->InnerVertices();
    double curr_value_sum = 0;
    double local_value_sum = 0;
    for(auto v : inner_vertices){
      local_value_sum += abs(app_->values_[v]- app_->last_values_[v]);
      app_->last_values_[v] = app_->values_[v];
    }
    communicator_.Sum(local_value_sum, curr_value_sum);
    return curr_value_sum < FLAGS_termcheck_threshold;
    */
  }

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> graph_;
  message_manager_t messages_;
  Communicator communicator_;
  std::thread send_th_;
  std::thread receive_th_;
  double last_delta_sum;
  double last_termcheck_ = 0.0;         // 上一次检测时间
  double termcheck_interval = 0.001;        // 设定的检测间隔
  double no_message_waite_time = 0.001;  // 当前没有接收到消息，等待时间
  std::vector<double> check_sum_values; // 统计每个线程最新的check信息


  CommSpec comm_spec_;

  class compare_priority {
  public:
      VertexArray<value_t, vid_t> &parent;
      
      compare_priority(VertexArray<value_t, vid_t>  &inparent): parent(inparent) {}
      
      bool operator()(const vid_t a, const vid_t b) {
        return abs(parent[Vertex<unsigned int>(a)]) > abs(parent[Vertex<unsigned int>(b)]);
      }
  };

};

}  // namespace grape

#endif  // GRAPE_WORKER_ASYNC_WORKER_H_
