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

#ifndef GRAPE_FRAGMENT_VC_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_VC_FRAGMENT_LOADER_H_

#include "grape/fragment/basic_vc_ds_fragment_loader.h"
#include "grape/fragment/basic_vc_fragment_loader.h"

namespace grape {

template <typename FRAG_T, typename IOADAPTOR_T = LocalIOAdaptor,
          typename LINE_PARSER_T =
              TSVLineParser<typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
                            typename FRAG_T::edata_t>>
class VCFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using vertices_t = typename fragment_t::vertices_t;

  using io_adaptor_t = IOADAPTOR_T;
  using line_parser_t = LINE_PARSER_T;

 public:
  explicit VCFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec) {}
  ~VCFragmentLoader() = default;

  std::shared_ptr<fragment_t> LoadFragment(int64_t vnum,
                                           const std::string& efile,
                                           const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);
    if (spec.deserialize) {
      fragment = std::make_shared<fragment_t>();
      fragment->template Deserialize<IOADAPTOR_T>(comm_spec_,
                                                  spec.deserialization_prefix);
      return fragment;
    }

    if (spec.single_scan) {
      fragment = loadFragmentSingleScan(vnum, efile, spec);
    } else {
      fragment = loadFragmentDoubleScan(vnum, efile, spec);
    }

    if (spec.serialize) {
      fragment->template Serialize<IOADAPTOR_T>(spec.serialization_prefix);
    }

    return fragment;
  }

 private:
  std::shared_ptr<fragment_t> loadFragmentSingleScan(
      int64_t vnum, const std::string& efile, const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);

    auto basic_fragment_loader =
        std::unique_ptr<BasicVCFragmentLoader<fragment_t>>(
            new BasicVCFragmentLoader<fragment_t>(comm_spec_, vnum,
                                                  spec.load_concurrency));

    auto io_adaptor = std::unique_ptr<io_adaptor_t>(new io_adaptor_t(efile));
    io_adaptor->SetPartialRead(comm_spec_.worker_id(), comm_spec_.worker_num());
    io_adaptor->Open();

    std::string line;
    edata_t e_data;
    oid_t src, dst;

    double t0 = -grape::GetCurrentTime();
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

      basic_fragment_loader->AddEdge(src, dst, e_data);
    }
    io_adaptor->Close();

    MPI_Barrier(comm_spec_.comm());
    t0 += grape::GetCurrentTime();
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished reading edges inputs, time: " << t0 << " s";
    }

    double t2 = -grape::GetCurrentTime();
    basic_fragment_loader->ConstructFragment(fragment);
    MPI_Barrier(comm_spec_.comm());
    t2 += grape::GetCurrentTime();

    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished constructing fragment, time: " << t2 << " s";
    }
#ifdef TRACKING_MEMORY
    VLOG(1) << "[frag-" << comm_spec_.fid() << "] after constructing fragment: "
            << MemoryTracker::GetInstance().GetMemoryUsageInfo();
#endif
    return fragment;
  }

  std::shared_ptr<fragment_t> loadFragmentDoubleScan(
      int64_t vnum, const std::string& efile, const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);

    auto basic_fragment_loader =
        std::unique_ptr<BasicVCDSFragmentLoader<fragment_t>>(
            new BasicVCDSFragmentLoader<fragment_t>(comm_spec_, vnum,
                                                    spec.load_concurrency));
    {
      auto io_adaptor = std::unique_ptr<io_adaptor_t>(new io_adaptor_t(efile));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();

      std::string line;
      edata_t e_data;
      oid_t src, dst;

      double t0 = -grape::GetCurrentTime();
      size_t lineNo = 0;
      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id()
                   << "][efile - scan 1 / 2] " << lineNo;
        }
        if (line.empty() || line[0] == '#')
          continue;

        try {
          line_parser_.LineParserForEFile(line, src, dst, e_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }

        basic_fragment_loader->RecordEdge(src, dst);
      }
      io_adaptor->Close();

      MPI_Barrier(comm_spec_.comm());
      t0 += grape::GetCurrentTime();
      if (comm_spec_.worker_id() == 0) {
        VLOG(1) << "finished first scan of edges inputs, time: " << t0 << " s";
      }
    }

    basic_fragment_loader->AllocateBuffers();
    MPI_Barrier(comm_spec_.comm());

    {
      auto io_adaptor = std::unique_ptr<io_adaptor_t>(new io_adaptor_t(efile));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();

      std::string line;
      edata_t e_data;
      oid_t src, dst;

      double t0 = -grape::GetCurrentTime();
      size_t lineNo = 0;
      while (io_adaptor->ReadLine(line)) {
        ++lineNo;
        if (lineNo % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id()
                   << "][efile - scan 2 / 2] " << lineNo;
        }
        if (line.empty() || line[0] == '#')
          continue;

        try {
          line_parser_.LineParserForEFile(line, src, dst, e_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }

        basic_fragment_loader->AddEdge(src, dst, e_data);
      }
      io_adaptor->Close();

      MPI_Barrier(comm_spec_.comm());
      t0 += grape::GetCurrentTime();
      if (comm_spec_.worker_id() == 0) {
        VLOG(1) << "finished second scan of edges inputs, time: " << t0 << " s";
      }
    }

    double t2 = -grape::GetCurrentTime();
    basic_fragment_loader->ConstructFragment(fragment);
    MPI_Barrier(comm_spec_.comm());
    t2 += grape::GetCurrentTime();

    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished constructing fragment, time: " << t2 << " s";
    }
#ifdef TRACKING_MEMORY
    VLOG(1) << "[frag-" << comm_spec_.fid() << "] after constructing fragment: "
            << MemoryTracker::GetInstance().GetMemoryUsageInfo();
#endif
    return fragment;
  }

  CommSpec comm_spec_;
  line_parser_t line_parser_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_VC_FRAGMENT_LOADER_H_
