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

#ifndef GRAPE_FRAGMENT_EV_FRAGMENT_MUTATOR_H_
#define GRAPE_FRAGMENT_EV_FRAGMENT_MUTATOR_H_

#include <grape/fragment/basic_fragment_mutator.h>
#include <grape/fragment/partitioner.h>
#include <grape/util.h>

namespace grape {
template <typename FRAG_T, typename IOADAPTOR_T>
class EVFragmentMutator {
  using fragment_t = FRAG_T;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vdata_t = typename FRAG_T::vdata_t;
  using edata_t = typename FRAG_T::edata_t;

 public:
  explicit EVFragmentMutator(const CommSpec& comm_spec)
      : comm_spec_(comm_spec), t0_(0), t1_(0) {}
  ~EVFragmentMutator() {
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "mutating graph: " << t0_ << " s + " << t1_ << " s";
    }
  }

  std::shared_ptr<fragment_t> MutateFragment(const std::string& efile,
                                             const std::string& vfile,
                                             std::shared_ptr<fragment_t> frag,
                                             bool directed) {
    VLOG(1) << "delta efile = " << efile << ", vfile = " << vfile;
    if (efile == "" && vfile == "") {
      return frag;
    }
    MPI_Barrier(comm_spec_.comm());
    t0_ -= GetCurrentTime();
    BasicFragmentMutator<fragment_t> mutator(comm_spec_, frag);
    mutator.Start();
    if (!vfile.empty() || !std::is_same<vdata_t, EmptyType>::value) {
      auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(vfile));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();
      std::string line;

      char type;
      oid_t id;
      vdata_t data;

      int count = 0;
      size_t lineNo = 0;
      std::istringstream istrm;
      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                   << lineNo;
        }
        if (line.empty() || line[0] == '#')
          continue;
        istrm.str(line);
        istrm >> type;
        if (type == 'a') {
          ++count;
          istrm >> id >> data;
          mutator.AddVertex(id, data);
        } else if (type == 'd') {
          istrm >> id;
          mutator.RemoveVertex(id);
        } else if (type == 'u') {
          istrm >> id >> data;
          mutator.UpdateVertex(id, data);
        }
      }
      VLOG(1) << "read vertices to add: " << count;
    }
    {
      auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(efile));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();

      std::string line;
      std::istringstream istrm;

      char type;
      oid_t src, dst;
      edata_t data;
      size_t lineNo = 0;
      int count = 0;
      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                   << lineNo;
        }
        istrm.str(line);
        istrm >> type;
        if (type == 'a') {
          ++count;
          istrm >> src >> dst >> data;
          mutator.AddEdge(src, dst, data);
        } else if (type == 'd') {
          istrm >> src >> dst;
          mutator.RemoveEdge(src, dst);
          if (!directed) {
            mutator.RemoveEdge(dst, src);
          }
        } else if (type == 'u') {
          istrm >> src >> dst >> data;
          mutator.UpdateEdge(src, dst, data);
          if (!directed) {
            mutator.UpdateEdge(dst, src, data);
          }
        }
      }
      VLOG(1) << "read edges to add: " << count;
    }
    MPI_Barrier(comm_spec_.comm());
    t0_ += GetCurrentTime();
    t1_ -= GetCurrentTime();
    auto ret = mutator.MutateFragment();
    MPI_Barrier(comm_spec_.comm());
    t1_ += GetCurrentTime();

    return ret;
  }

 private:
  CommSpec comm_spec_;
  double t0_;
  double t1_;
};
}  // namespace grape

#endif  // GRAPE_FRAGMENT_EV_FRAGMENT_MUTATOR_H_
