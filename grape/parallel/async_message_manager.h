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

#ifndef GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_

#include <memory>
#include <utility>
#include <vector>
#include <queue>

#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/worker/comm_spec.h"

namespace grape {
enum ManagerStatus { STOPPED, RUNNING };
// const int ASYNC_TAG = 0xff;  // tag = 0 is used in communicator.h
enum MessageTag { Flag_stop=1, Flag_start=2, Flag_check=4, Flag_any_tag=0xff};

class RPCRequest {
  public:
    int tag;       // 发送或接收消息的flag     
    int fid;       // id
    InArchive arc; // data
    RPCRequest(){}
    RPCRequest(int id, InArchive&& arcr, int targe){
      tag = targe;
      fid = id;
      arc = std::move(arcr);
    }
    RPCRequest(RPCRequest&& rpc){ // 不要const
      tag = rpc.tag;
      fid = rpc.fid;
      arc = std::move(rpc.arc);
    }
    RPCRequest& operator=(RPCRequest&& rpc) {
      tag = rpc.tag;
      fid = rpc.fid;
      arc = std::move(rpc.arc);
      return *this;
    }
    ~RPCRequest(){}
};


/**
 * @brief Default message manager.
 *
 * The send and recv methods are not thread-safe.
 */
class AsyncMessageManager {
 public:
  AsyncMessageManager() : comm_(NULL_COMM) {}

  ~AsyncMessageManager() {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Inherit
   */
  void Init(MPI_Comm comm) {
    MPI_Comm_dup(comm, &comm_); //将现有的通信器及其所有缓存的信息复制

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    to_recv_.SetProducerNum(1);
    sent_size_ = 0;
    manager_status_.store(ManagerStatus::STOPPED);
  }

  static void Sleep(double t) {
    timespec req;
    req.tv_sec = (int) t;
    req.tv_nsec = (int64_t)(1e9 * (t - (int64_t) t));
    nanosleep(&req, NULL);
  }

  /**
   * @brief Inherit
   */
  void Start() {
    manager_status_.store(ManagerStatus::RUNNING);
    // /*
    //清理reqs_的线程
    cleaner_th_ = std::thread([this]() {
      // Sleep(3);
      // 测试
      int finish_num = 0;
      while (manager_status_.load() != ManagerStatus::STOPPED) {
        // auto erase_begin = GetCurrentTime();
        if (manager_status_.load() == ManagerStatus::RUNNING) {

          // std::unique_lock<std::mutex> lk(send_mux_); // ？？？导致不能发送请求了
          
          { // 锁reqs_
              std::unique_lock<std::mutex> lk(reqs_mux_1);
              // std::unique_lock<std::mutex> lk(send_mux_);
              for (auto it = reqs_.begin(); it != reqs_.end() && manager_status_.load() == ManagerStatus::RUNNING;) {
                int flag = 0;
                MPI_Request req = it->first;
                MPI_Test(&req, &flag, MPI_STATUSES_IGNORE);  
                if (flag) {
	                finish_num++;
                  it = reqs_.erase(it);
                } else {
                  it++;
                }
              }
          }
          { // 锁reqs_2
              std::unique_lock<std::mutex> lk(reqs_mux_2);
              // std::unique_lock<std::mutex> lk(send_mux_);
              for (auto it = reqs_2.begin(); it != reqs_2.end() && manager_status_.load() == ManagerStatus::RUNNING;) {
                int flag = 0;
                MPI_Request req = it->first;
                MPI_Test(&req, &flag, MPI_STATUSES_IGNORE);  
                if (flag) {
                  finish_num++;
                  it = reqs_2.erase(it);
                } else {
                  it++;
                }
              }
          }

            // auto it = reqs_.begin();
            // while(it != reqs_.end()) {
            //   std::unique_lock<std::mutex> lk(send_mux_); 
            //   int flag = 0;
            //   MPI_Request req = it->first;
            //   // lk.unlock();
            //   MPI_Test(&req, &flag, MPI_STATUSES_IGNORE);
            //   // lk.lock();
            //   if (flag) {
            //     it = reqs_.erase(it);
            //     continue;
            //   }
            //   it++;
            // }

        }
        //        std::this_thread::yield();
        Sleep(0.1);
        //        LOG(INFO) << "Pending size: " << reqs_.size()
        //                  << " erase time: " << GetCurrentTime() -
        //                  erase_begin;
      }
      LOG(INFO) << "清理线程结束！！ 测试完成数量： " << finish_num;
    });
    // */

    // 测试
    // 记录实际发送消息的数量
    int real_send_num = 0;

    do {
      // 消息的实际接收
      // MPI_Status status;
      // int flag;
      // MPI_Iprobe(MPI_ANY_SOURCE, ASYNC_TAG, comm_, &flag, &status); //非阻塞型探测，无论是否有一个符合条件的消息到达，立即返回。有flag=true；否则flag=false
      //
      // if (flag) {
      //   int length;
      //   auto src_worker = status.MPI_SOURCE;
      //   MPI_Get_count(&status, MPI_CHAR, &length);
      //
      //   CHECK_GT(length, 0);
      //   OutArchive arc(length);
      //   MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, ASYNC_TAG,
      //            comm_, MPI_STATUS_IGNORE);
      //   to_recv_.Put(std::move(arc));
      // }

      // 消息的实际发送
      // 1.
      // std::unique_lock<std::mutex> lk(send_mux_);
      // {
      //   std::unique_lock<std::mutex> lk(send_mux_, std::try_to_lock);
      //   if (lk.owns_lock() == true) {
      //     for (auto& e : to_send_) {
      //       InArchive arc(std::move(e.second));
      //       MPI_Request req;
      //       MPI_Isend(arc.GetBuffer(), arc.GetSize(), MPI_CHAR,
      //                 comm_spec_.FragToWorker(e.first), ASYNC_TAG, comm_, &req);
      //       reqs_.emplace(req, std::move(arc));
      //     }
      //     to_send_.clear();
      //     continue;
      //   }
      // }
      // {
      //   std::unique_lock<std::mutex> lk(send_mux_, std::try_to_lock);
      //   if (lk.owns_lock() == true) {
      //     for (auto& e : to_send_) {
      //       InArchive arc(std::move(e.second));
      //       MPI_Request req;
      //       MPI_Isend(arc.GetBuffer(), arc.GetSize(), MPI_CHAR,
      //                 comm_spec_.FragToWorker(e.first), ASYNC_TAG, comm_, &req);
      //       reqs_2.emplace(req, std::move(arc));
      //     }
      //     to_send_.clear();
      //   }
      // }

      
      // 2.缩小临界区
      while(manager_status_.load() == ManagerStatus::RUNNING && !to_send_.empty()){
        // std::unique_lock<std::mutex> lk(send_mux_);
        {
          std::unique_lock<std::mutex> lk(send_mux_, std::try_to_lock); // 用于尝试锁to_send_
          if (lk.owns_lock() == true) {
            // auto& e = to_send_.back();
            auto& e = to_send_.front();
            InArchive arc(std::move(e.arc)); 
            auto dest = e.fid;
            int tag = e.tag;
            to_send_.pop(); // 必须要在加入arc和保存了e.first之后才能清除，不然空间释放了引用就无效
            lk.unlock(); // 解锁

            MPI_Request req;
            // LOG(INFO) << "tag: " <<  tag << MessageTag::Flag_any_tag << ", " << arc.GetSize();
            MPI_Isend(arc.GetBuffer(), arc.GetSize(), MPI_CHAR,
                      comm_spec_.FragToWorker(dest), tag, comm_, &req);   
            // LOG(INFO) << "发送目的地： " << comm_spec_.FragToWorker(dest);
            real_send_num++; // 记录消息发送数量
            // lk.lock(); // 锁reqs_
            // reqs_.emplace(req, std::move(arc)); // 是否需要单独放一个锁？？？
            // continue;

            {
              std::unique_lock<std::mutex> lk1(reqs_mux_1, std::try_to_lock); // 尝试锁reqs_
              if(lk1.owns_lock() == true){
                reqs_.emplace(req, std::move(arc));  
                continue;
              }
            }
            {
              std::unique_lock<std::mutex> lk2(reqs_mux_2, std::try_to_lock); // 尝试锁reqs_2
              if(lk2.owns_lock() == true){
                reqs_2.emplace(req, std::move(arc));  
                continue;
              }
            }
            // 如果上面都失败了,只能等,随便等一个即可
            {
              std::unique_lock<std::mutex> lk(reqs_mux_1); // 锁reqs_
              reqs_.emplace(req, std::move(arc));
              printf("lk1,lk2都获取锁失败后重新获取！\n");
            }
            
          }
        }
        
        // {
        //   std::unique_lock<std::mutex> lk(send_mux_, std::try_to_lock);
        //   if (lk.owns_lock() == true) {
        //     // auto& e = to_send_.back();
        //     auto& e = to_send_.back();
        //     InArchive arc(std::move(e.second)); 
        //     auto dest = e.first;
        //     to_send_.pop_back(); // 必须要在加入arc和保存了e.first之后才能清除，不然空间释放了引用就无效
        //     lk.unlock(); // 提前解锁
        //
        //     MPI_Request req;
        //     MPI_Isend(arc.GetBuffer(), arc.GetSize(), MPI_CHAR,
        //               comm_spec_.FragToWorker(dest), ASYNC_TAG, comm_, &req);   
        //          
        //     lk.lock(); // 锁reqs_
        //     reqs_2.emplace(req, std::move(arc));
        //     continue;
        //   }
        // }
      }

      // 3.直接调用发送，不缓存到to_send_

      

    } while (manager_status_.load() != ManagerStatus::STOPPED);

    //调试
    LOG(INFO) << "当前to_send_的大小: " << to_send_.size() << ", 当前reqs_的大小: " << reqs_.size() + reqs_2.size();

    LOG(INFO) << "AsyncMessageManager has benn stopped. 实际发送消息： real_send_num = " << real_send_num;
  }

  bool IsSendFinish(){
    return reqs_.size() + reqs_2.size() == 0 && to_send_.size() == 0; // 发完并被接收完
  }

  // 接收线程
  void ReceiveThread(){
    manager_status_.store(ManagerStatus::RUNNING); // 不然太快，读到的是初始的STOPPED
    int real_rece_num = 0;
    int try_rece_num = 0;
    while (manager_status_.load() != ManagerStatus::STOPPED) {
      // auto erase_begin = GetCurrentTime();
      if (manager_status_.load() == ManagerStatus::RUNNING) {
        // 消息的实际接收
        MPI_Status status;
        int flag;
        /*
        { // 优先探测是否有stop消息
          MPI_Iprobe(MPI_ANY_SOURCE, 1, comm_, &flag, &status);
          if(flag){
            int length;
            fid_t src_worker = status.MPI_SOURCE;
            auto tag = status.MPI_TAG;
            if(tag == 1){
              // manager_status_.store(ManagerStatus::STOPPED); // 直接停止，导致其他线程在jion之前停止
              MPI_Get_count(&status, MPI_CHAR, &length);
              CHECK_GT(length, 0);
              OutArchive arc(length);
              MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, tag,
                      comm_, MPI_STATUS_IGNORE);
              real_rece_num++;
              is_get_stop = true; // 标记已经收到stop // 0号线程，标记停止
              LOG(INFO) << "收到stop消息";
            }
          }
        }
        { // 探测是否有check消息
          MPI_Iprobe(MPI_ANY_SOURCE, 4, comm_, &flag, &status);
          if(flag){
            int length;
            fid_t src_worker = status.MPI_SOURCE;
            auto tag = status.MPI_TAG;
            if(tag == 4){
              MPI_Get_count(&status, MPI_CHAR, &length);
              CHECK_GT(length, 0);
              OutArchive arc(length);
              MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, tag,
                      comm_, MPI_STATUS_IGNORE);
              real_rece_num++;
              std::unique_lock<std::mutex> lk(recv_check_mux_);
              to_recv_check_.emplace(src_worker, std::move(arc));
            }
          }
        }
        */
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &flag, &status); //非阻塞型探测，无论是否有一个符合条件的消息到达，立即返回。有flag=true；否则flag=false

        if (flag) {
          int length;
          fid_t src_worker = status.MPI_SOURCE;
          auto tag = status.MPI_TAG;
          // printf("ta=%d\n", tag);
          if(tag == 1){
            // manager_status_.store(ManagerStatus::STOPPED); // 直接停止，导致其他线程在jion之前停止
            MPI_Get_count(&status, MPI_CHAR, &length);
            CHECK_GT(length, 0);
            OutArchive arc(length);
            MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, tag,
                    comm_, MPI_STATUS_IGNORE);
            real_rece_num++;
            is_get_stop = true; // 标记已经收到stop // 0号线程，标记停止
            LOG(INFO) << "收到stop消息";
          }
          else if (tag == 4)
          {
            MPI_Get_count(&status, MPI_CHAR, &length);
            CHECK_GT(length, 0);
            OutArchive arc(length);
            MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, tag,
                    comm_, MPI_STATUS_IGNORE);
            real_rece_num++;
            std::unique_lock<std::mutex> lk(recv_check_mux_);
            to_recv_check_.emplace(src_worker, std::move(arc));
            // printf("size:: %d\n", to_recv_check_.size());
          }
          else{   
            MPI_Get_count(&status, MPI_CHAR, &length);
            CHECK_GT(length, 0);
            OutArchive arc(length);
            MPI_Recv(arc.GetBuffer(), length, MPI_CHAR, src_worker, MPI_ANY_TAG,
                    comm_, MPI_STATUS_IGNORE);
            real_rece_num++;
            to_recv_.Put(std::move(arc));
          }
        }
        // try_rece_num++;

        // else{
        //   Sleep(0.00001);
        // }  
      }
    }
    LOG(INFO) << "接收线程完成！！！实际接收消息：real_rece_num=" << real_rece_num << ", status: " << manager_status_.load() << ", request_size: " << reqs_2.size() + reqs_.size() << ", 尝试接收了多少次: " << try_rece_num;
  }
 
  // 标记停止
  void Set_is_get_Stop(){
    is_get_stop = true;
  }

  bool Get_is_get_Stop(){
    return is_get_stop;
  }

  void Stop() {
    if (manager_status_.load() == ManagerStatus::RUNNING) {
      manager_status_.store(ManagerStatus::STOPPED);
    }
    is_get_stop = true;
    cleaner_th_.join();
  }

  void Stop2() {
    if (manager_status_.load() == ManagerStatus::RUNNING) {
      manager_status_.store(ManagerStatus::STOPPED);
    }
  }

  bool status() {
    if (manager_status_.load() == ManagerStatus::STOPPED) {
      return true;
    }
    return false;
  }

  /**
   * @brief Inherit
   */
  void Finalize() {
    MPI_Comm_free(&comm_);
    comm_ = NULL_COMM;
  }

  size_t GetMsgSize() const { return sent_size_; }

  /**
   * @brief Send message to a fragment.
   *
   * @tparam MESSAGE_T Message type.
   * @param dst_fid Destination fragment id.
   * @param msg
   */
  template <typename MESSAGE_T>
  inline void SendToFragment(fid_t dst_fid, const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    InArchive archive;

    archive << msg;
    // send(dst_fid, std::move(archive), tag);
    MPI_Request req;
    MPI_Isend(archive.GetBuffer(), archive.GetSize(), MPI_CHAR,
                      comm_spec_.FragToWorker(dst_fid), tag, comm_, &req);
    reqs_check.emplace(req, std::move(archive)); // 暂未清理 reqs_check
  }

    /**
   * @brief Send message to a fragment.
   *
   * @tparam MESSAGE_T Message type.
   * @param dst_fid Destination fragment id.
   * @param msg
   */
  template <typename MESSAGE_T>
  inline void SendToFragment_block(fid_t dst_fid, const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    InArchive archive;

    archive << msg;
    // send(dst_fid, std::move(archive), tag);
    MPI_Send(archive.GetBuffer(), archive.GetSize(), MPI_CHAR,
                      comm_spec_.FragToWorker(dst_fid), tag, comm_);
  }
 

  /**
   * @brief Communication by synchronizing the manager_status_ on outer
   * vertices, for edge-cut fragments.
   *
   * Assume a fragment F_1, a crossing edge a->b' in F_1 and a is an inner
   * vertex in F_1. This function invoked on F_1 send manager_status_ on b' to b
   * on F_2, where b is an inner vertex.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                     const typename GRAPH_T::vertex_t& v,
                                     const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    fid_t fid = frag.GetFragId(v);
    InArchive archive;

    archive << frag.GetOuterVertexGid(v) << msg;
    send(fid, std::move(archive), tag);
  }

  /**
   * @brief Communication via a crossing edge a<-c. It sends message
   * from a to c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughIEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    DestList dsts = frag.IEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive), tag);
    }
  }

  /**
   * @brief Communication via a crossing edge a->b. It sends message
   * from a to b.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughOEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    DestList dsts = frag.OEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive), tag);
    }
  }

  /**
   * @brief Communication via crossing edges a->b and a<-c. It sends message
   * from a to b and c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughEdges(const GRAPH_T& frag,
                                  const typename GRAPH_T::vertex_t& v,
                                  const MESSAGE_T& msg, int tag=MessageTag::Flag_any_tag) {
    DestList dsts = frag.IOEDests(v);
    fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      InArchive archive;

      archive << gid << msg;
      send(fid, std::move(archive), tag);
    }
  }

  /**
   * @brief Get a message from message buffer.
   *
   * @tparam MESSAGE_T
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename MESSAGE_T>
  inline bool GetMessage(MESSAGE_T& msg) {
    if (to_recv_.Size() == 0)
      return false;
    OutArchive arc;
    to_recv_.Get(arc);

    arc >> msg;
    return true;
  }

  /**
   * @brief Get a message and its target vertex from message buffer.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline bool GetMessage(const GRAPH_T& frag, typename GRAPH_T::vertex_t& v,
                         MESSAGE_T& msg) {
    OutArchive arc;
    if (to_recv_.Size() == 0 || !to_recv_.Get(arc))
      return false;

    typename GRAPH_T::vid_t gid;
    arc >> gid >> msg;
    // printf("gid: %d, ", gid);
    CHECK(frag.Gid2Vertex(gid, v));
    return true;
  }

    /**
   * @brief Get a check message and its check value.
   *
   * @tparam Fid_t
   * @tparam MESSAGE_T
   * @param frag
   * @param v
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename Fid_t, typename MESSAGE_T>
  inline bool GetMessage(Fid_t& v,  MESSAGE_T& msg) {
    OutArchive arc;
    std::unique_lock<std::mutex> lk(recv_check_mux_);
    if (to_recv_check_.size()== 0)
      return false;
    auto ck = to_recv_check_.front();
    // lk.unlock();
    v = ck.first;
    arc = std::move(ck.second);
    to_recv_check_.pop();
    arc >> msg;
    return true;
  }

 private:
  inline void send(fid_t fid, InArchive&& arc, int tag=MessageTag::Flag_any_tag) {
    if (arc.Empty()) {
      return;
    }

    if (fid == fid_) {  // self message
      OutArchive tmp(std::move(arc));
      to_recv_.Put(tmp);
    } else {
      sent_size_ += arc.GetSize();
      CHECK_GT(arc.GetSize(), 0);

      std::unique_lock<std::mutex> lk(send_mux_);
      // to_send_.emplace(fid, std::move(arc));  

      RPCRequest r(fid, std::move(arc), tag);
      to_send_.emplace(std::move(r));
    }
  }

  // std::vector<std::pair<fid_t, InArchive>> to_send_;
  std::queue<RPCRequest> to_send_;
  BlockingQueue<OutArchive> to_recv_{};
  std::queue<std::pair<fid_t, OutArchive>> to_recv_check_; // 存放0号进程收到的各个进程提交的检查消息

  std::mutex send_mux_;
  std::mutex reqs_mux_1;  // 用于锁定reqs_
  std::mutex reqs_mux_2;  // 用于锁定reqs_2
  std::mutex check_mux_;  // 锁ischeck
  std::mutex recv_check_mux_;  // 锁ischeck

  std::unordered_map<MPI_Request, InArchive> reqs_;     // 暂存请求，不然离开作用域就释放了
  std::unordered_map<MPI_Request, InArchive> reqs_2;    // 暂存请求，不然离开作用域就释放了
  std::unordered_map<MPI_Request, InArchive> reqs_check;    // 暂存check\stop请求

  MPI_Comm comm_;

  fid_t fid_{}; //initializer_list
  fid_t fnum_{};
  CommSpec comm_spec_;
  std::thread cleaner_th_;  // 清理线程
  bool is_get_stop = false; //表示是否需要检测

  size_t sent_size_{};
  std::atomic_int manager_status_{};
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_ASYNC_MESSAGE_MANAGER_H_
