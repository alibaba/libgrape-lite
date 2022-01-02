
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_TIME_TABLE_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_TIME_TABLE_H_

#include "VariadicTable.h"
#include "grape/worker/comm_spec.h"

namespace grape_gpu {

template <typename data_t = double>
class TimeTable {
 public:
  explicit TimeTable(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec),
        n_worker_(comm_spec.worker_num()),
        n_row_(std::numeric_limits<size_t>::max()) {}

  void AddColumn(const std::string& col_name, const std::vector<data_t>& data) {
    if (n_row_ == std::numeric_limits<size_t>::max()) {
      n_row_ = data.size();
    } else {
      CHECK_EQ(n_row_, data.size()) << "Col name: " << col_name;
    }

    std::vector<data_t> all_data(n_row_ * n_worker_);

    if (std::is_same<data_t, double>::value) {
      MPI_Gather(data.data(), n_row_, MPI_DOUBLE, all_data.data(), n_row_,
                 MPI_DOUBLE, 0, comm_spec_.comm());
    } else if (std::is_same<data_t, size_t>::value) {
      MPI_Gather(data.data(), n_row_, my_MPI_SIZE_T, all_data.data(), n_row_,
                 my_MPI_SIZE_T, 0, comm_spec_.comm());
    } else if (std::is_same<data_t, int>::value) {
      MPI_Gather(data.data(), n_row_, MPI_INT, all_data.data(), n_row_, MPI_INT,
                 0, comm_spec_.comm());
    } else if (std::is_same<data_t, uint32_t>::value) {
      MPI_Gather(data.data(), n_row_, MPI_UINT32_T, all_data.data(), n_row_,
                 MPI_UINT32_T, 0, comm_spec_.comm());
    } else if (std::is_same<data_t, uint64_t>::value) {
      MPI_Gather(data.data(), n_row_, MPI_UINT64_T, all_data.data(), n_row_,
                 MPI_UINT64_T, 0, comm_spec_.comm());
    }

    if (comm_spec_.worker_id() == 0) {
      col_names_.push_back(col_name);
      cols_.push_back(all_data);

      std::vector<data_t> gb_worker;

      for (int worker_id = 0; worker_id < n_worker_; worker_id++) {
        data_t total = 0;
        for (int round = 0; round < n_row_; round++) {
          auto idx = worker_id * n_row_ + round;

          total += all_data[idx];
        }
        gb_worker.push_back(total);
      }
      summary_.push_back(gb_worker);
    }
  }

  void Print() {
    if (comm_spec_.worker_id() == 0) {
      auto headers = col_names_;
      headers.emplace_back("Total");
      headers.insert(headers.begin(), {"Round", "Worker"});

      switch (col_names_.size()) {
      case 1: {
        using vt_t = VariadicTable<int, int, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], total);
          }
        });
        break;
      }
      case 2: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx], total);
          }
        });
        break;
      }
      case 3: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], total);
          }
        });
        break;
      }
      case 4: {
        using vt_t =
            VariadicTable<int, int, data_t, data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], total);
          }
        });
        break;
      }
      case 5: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t, data_t,
                                   data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx], total);
          }
        });
        break;
      }
      case 6: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t, data_t,
                                   data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id],
                      summary_[5][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx],
                      cols_[5][idx], total);
          }
        });
        break;
      }
      case 7: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t, data_t,
                                   data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id],
                      summary_[5][worker_id], summary_[6][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx],
                      cols_[5][idx], cols_[6][idx], total);
          }
        });
        break;
      }
      case 8: {
        using vt_t = VariadicTable<int, int, data_t, data_t, data_t, data_t,
                                   data_t, data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id],
                      summary_[5][worker_id], summary_[6][worker_id],
                      summary_[7][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx],
                      cols_[5][idx], cols_[6][idx], cols_[7][idx], total);
          }
        });
        break;
      }
      case 9: {
        using vt_t =
            VariadicTable<int, int, data_t, data_t, data_t, data_t, data_t,
                          data_t, data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id],
                      summary_[5][worker_id], summary_[6][worker_id],
                      summary_[7][worker_id], summary_[8][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx],
                      cols_[5][idx], cols_[6][idx], cols_[7][idx],
                      cols_[8][idx], total);
          }
        });
        break;
      }
      case 10: {
        using vt_t =
            VariadicTable<int, int, data_t, data_t, data_t, data_t, data_t,
                          data_t, data_t, data_t, data_t, data_t, data_t>;
        vt_t vt(headers);

        MakeTable(vt, [&, this](int round, int worker_id, int idx) {
          data_t total = 0;

          if (round == -1) {
            for (int col = 0; col < col_names_.size(); col++) {
              total += summary_[col][worker_id];
            }
            vt.addRow(round, worker_id, summary_[0][worker_id],
                      summary_[1][worker_id], summary_[2][worker_id],
                      summary_[3][worker_id], summary_[4][worker_id],
                      summary_[5][worker_id], summary_[6][worker_id],
                      summary_[7][worker_id], summary_[8][worker_id],
                      summary_[9][worker_id], total);
          } else {
            for (int col = 0; col < col_names_.size(); col++) {
              total += cols_[col][idx];
            }
            vt.addRow(round, worker_id, cols_[0][idx], cols_[1][idx],
                      cols_[2][idx], cols_[3][idx], cols_[4][idx],
                      cols_[5][idx], cols_[6][idx], cols_[7][idx],
                      cols_[8][idx], cols_[9][idx], total);
          }
        });
        break;
      }
      default:
        CHECK(false);
      }
    }
  }

 private:
  grape::CommSpec comm_spec_;
  int n_worker_;
  size_t n_row_;
  std::vector<std::string> col_names_;
  std::vector<std::vector<data_t>> cols_;
  std::vector<std::vector<data_t>> summary_;

  template <class TABLE, class ADD_ROW>
  void MakeTable(TABLE& vt, ADD_ROW add_row) {
    for (int round = 0; round < n_row_; round++) {
      for (int worker_id = 0; worker_id < n_worker_; worker_id++) {
        auto idx = worker_id * n_row_ + round;

        add_row(round, worker_id, idx);
      }
    }

    for (int worker_id = 0; worker_id < n_worker_; worker_id++) {
      add_row(-1, worker_id, -1);
    }

    std::vector<int> prec{0, 0};
    std::vector<VariadicTableColumnFormat> format{
        VariadicTableColumnFormat::AUTO,
        VariadicTableColumnFormat::AUTO,
    };

    if (std::is_same<data_t, float>::value ||
        std::is_same<data_t, double>::value) {
      for (int i = 0; i <= col_names_.size(); i++) {
        prec.push_back(3);
        format.push_back(VariadicTableColumnFormat::FIXED);
      }
    }

    vt.setColumnPrecision(prec);
    vt.setColumnFormat(format);

    std::stringstream ss;
    ss << "Stat: " << std::endl;
    vt.print(ss);
    std::cout << ss.str() << std::endl;
  }
};

}  // namespace grape_gpu

#endif  // GRAPEGPU_GRAPE_GPU_UTILS_TIME_TABLE_H_
