#include "./solver.h"
#include "ortools/linear_solver/linear_solver.h"
#include <chrono>

namespace grape_gpu {
std::vector<std::vector<size_t>>
ILP_solver(std::vector<std::vector<double>>& C, std::vector<size_t>& L){
  namespace ort=operations_research;

  auto start = std::chrono::system_clock::now();
  // Data
  int n = L.size(); // 8
  std::vector<std::vector<size_t>> tr(n, std::vector<size_t>(n, 0));
  //for(int i=0; i<n; ++i){
  //  for(int j=0; j<n; ++j){
  //    if(C[i][j] == 0) C[i][j]=8;
  //  }
  //}
  auto i_inf = std::numeric_limits<size_t>::max();
  size_t max_ = i_inf;
  for(auto l: L) max_ = std::max(max_,l);
  for(int i=0; i<n; ++i){
    for(int j=0; j<n; ++j){
      C[i][j] = C[i][j];
    }
  }

  // Solver
  std::unique_ptr<ort::MPSolver> solver(ort::MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    //LOG(WARNING) << "SCIP solver unavailable.";
    exit(0);
  }

  // Variables
  // x[i][j] is an array of 0-m variables
  // auxiliary variable z to replace inner max
  std::vector<std::vector<const ort::MPVariable*>> x(n, std::vector<const ort::MPVariable*>(n));
  const ort::MPVariable* z;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      x[i][j] = solver->MakeIntVar(0, max_, "");
    }
  }
  z = solver->MakeIntVar(0, i_inf, "");//FIXME: IntVar?

  // Constraints
  // the cost of each worker should less than the auxiliary variables z.
  for (int j = 0; j < n; ++j) {
    ort::LinearExpr worker_cost;
    worker_cost += z;
    for (int i = 0; i < n; ++i) {
      worker_cost += -C[i][j] * ort::LinearExpr(x[i][j]);
    }
    solver->MakeRowConstraint(worker_cost >= 0.0);
  }
  // Each task is assigned to exactly one worker
  for (int i = 0; i < n; ++i) {
    ort::LinearExpr task_sum;
    for (int j = 0; j < n; ++j) {
      task_sum += x[i][j];
    }
    solver->MakeRowConstraint(task_sum == L[i]);
  }

  // Objective.
  ort::MPObjective* const objective = solver->MutableObjective();
  objective->SetCoefficient(z, 1.0);
  objective->SetMinimization();

  // Solve
  const ort::MPSolver::ResultStatus result_status = solver->Solve();
  if (result_status != ort::MPSolver::OPTIMAL &&
      result_status != ort::MPSolver::FEASIBLE) {
    //LOG(FATAL) << "No solution found.";
    exit(1);
  }

  //LOG(INFO) << "Total cost = " << objective->Value() << "\n\n";

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      tr[i][j] = (size_t) x[i][j]->solution_value();
    }
  }

  //round up
  for (int i = 0; i < n; ++i) {
    size_t sum_ = 0;
    for (int j = 0; j < n; ++j) {
      sum_ += tr[i][j];
      //round up
      assert(tr[i][j] >=0);
    }
    tr[i][i] += sum_-L[i];
    assert(tr[i][i]>=0);
  }
  auto end = std::chrono::system_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  std::cout << "real time: " << double(duration.count()) * 1000 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
  return  tr;
}
}
