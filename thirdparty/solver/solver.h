#ifndef GRAPE_GPU_UTILS_SOLVER_H_
#define GRAPE_GPU_UTILS_SOLVER_H_

#include <assert.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
//#include "ortools/linear_solver/linear_solver.h"

namespace grape_gpu {


std::vector<std::vector<size_t>> ILP_solver(std::vector<std::vector<double>>& C, std::vector<size_t>& L);


template <typename N_WORK_T>
std::vector<std::vector<N_WORK_T>> ADP_solver(
    std::vector<std::vector<double>>& bw, std::vector<N_WORK_T>& load,
    double k) {
  // samll dense graph;
  typedef struct status_t {
    size_t cap{};
    double cost{};
    status_t(size_t a, double b) : cap(a), cost(b) {}
    status_t() = default;
  } status_t;

  auto i_inf = std::numeric_limits<N_WORK_T>::max();
  int onv = load.size();
  auto nv = onv + 2;

  std::vector<N_WORK_T> target(onv, 0);
  std::vector<double> weight(onv, 1);
  weight[3] = 1.1;
  weight[2] = 1;
  weight[1] = 1;
  weight[0] = 0.9;
  

  int64_t sum = 0, target_sum=0;
  double wsum = 0;
  for (auto val : load) sum += val;
  for (auto val : weight) wsum += val;
  for (int i = 0; i < onv; ++i) {
    target[i] = sum * weight[i] / wsum ;
  }
  for(auto val : target) target_sum += val;
  for(int i=0; target_sum < sum; ++i){
    target[i] ++;
    target_sum ++;
  }
  for(int i=0; target_sum > sum; ++i){
    target[i] --;
    target_sum --;
  }
  auto s = onv;
  auto e = onv + 1;
  auto sup = sum + 1;
  double inf = i_inf;

  std::vector<std::vector<status_t>> g(nv, std::vector<status_t>(nv));
  std::vector<std::vector<N_WORK_T>> tr(onv, std::vector<N_WORK_T>(onv, 0));

  auto reset = [&](){
    for(int i = 0; i< nv; ++i){
      for(int j = 0; j < nv; ++j){
        g[i][j].cap = 0;
        g[i][j].cost = 0;
      }
    }

    for (int i = 0; i < onv; ++i) {
      if (load[i] > target[i])
        g[s][i].cap = load[i] - target[i];
    }
  
    for (int i = 0; i < onv; ++i) {
      if (load[i] < target[i])
        g[i][e].cap = target[i] - load[i];
    }
  
    for (int i = 0; i < onv; ++i) {
      for (int j = 0; j < onv; ++j) {
        if (load[i] > target[i] && load[j] < target[j]) {
          double cost = bw[i][j] == 0 ? inf : (1 / bw[i][j]);
          g[i][j].cap = sup;
          g[i][j].cost = cost;
          g[j][i].cap = 0;
          g[j][i].cost = -cost;
        }
      }
    }
  };

  auto spfa = [&](){
    std::vector<int> vis(nv, 0);
    std::vector<int> pre(nv, -1);
    std::vector<double> dis(nv, inf);
    std::queue<int> q;
    q.push(s);
    vis[s] = 1;
    dis[s] = 0;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      vis[u] = 0;
      for (int v = 0; v < nv; ++v) {
        if (g[u][v].cap == 0)
          continue;
        double cost = g[u][v].cost;
        if (dis[u] != inf && dis[v] - dis[u] - cost > 1e-7) {
          dis[v] = dis[u] + cost;
          pre[v] = u;
          if (vis[v] == 0) {
            vis[v] = 1;
            q.push(v);
          }
        }
      }
    }

    if (dis[e] == inf)
      return false;
    else {
      auto flow = i_inf;
      for (int c = e; c != s; c = pre[c]) {
        flow = std::min(flow, g[pre[c]][c].cap);
      }
      for (int c = e; c != s; c = pre[c]) {
        g[pre[c]][c].cap -= flow;
        g[c][pre[c]].cap += flow;
      }
      return true;
    }
  };

  auto adjust= [&]() -> double{
    for (int i = 0; i < onv; ++i) {
      for (int j = 0; j < onv; ++j) {
        tr[i][j] = 0;
        if (load[i] > target[i] && load[j] < target[j]) {
          tr[i][j] = sup - g[i][j].cap;
        }
      }
    }
    int min_i=0, max_i=0;
    double min_=i_inf, max_=0;
    for(int i=0; i<onv; ++i){
      double sum = 0;
      N_WORK_T local = load[i];
      for(int j=0; j<onv; ++j){
        if(i==j || bw[j][i] == 0) continue;
        local -= tr[i][j];
        sum += tr[j][i]*tr[j][i] / bw[j][i] / weight[i];
      }
      sum += local*local / k / weight[i];
      if(sum > max_) {
        max_i = i; 
        max_  = sum;
      }
      if(sum < min_) {
        min_i = i;
        min_  = sum;
      }
    }
    double p = (max_-min_)/ max_ /2;
    if(min_i != max_i) {
      N_WORK_T change = target[max_i] * p;
      target[min_i]  += change;
      target[max_i]  -= change;
    }
    return max_ - min_;
  };

  auto check = [&]() {
    N_WORK_T sum = 0, target_sum = 0;
    for(int i=0; i<onv; ++i) sum += load[i];
    for(int i=0; i<onv; ++i) target_sum += target[i];
    assert(sum == target_sum);
  };
  
  int T = 1;
  while (T--){
    reset();
    while(spfa());
    double score = adjust();
    //std::cout << score << endl;
    check();
  }
  return tr;
}

template <typename N_WORK_T>
std::vector<std::vector<N_WORK_T>> control_solver(
    std::vector<std::vector<double>>& bw, std::vector<N_WORK_T>& load) {
  int nv = load.size();
  std::vector<std::vector<N_WORK_T>> tr(nv);
  for(int i=0; i<nv; ++i){
    tr[i].resize(nv, 0);
    tr[i][i] = load[i];
  }
  return tr;
}

template <typename N_WORK_T>
std::vector<std::vector<N_WORK_T>> test_solver(
    std::vector<std::vector<double>>& bw, std::vector<N_WORK_T>& load) {
  int nv = load.size();
  std::vector<std::vector<N_WORK_T>> tr(nv);
  for(int i=0; i<nv; ++i){
    tr[i].resize(nv, 0);
    for(int j=0; j<nv; ++j) {
       tr[i][j] = load[i] / nv + (j < load[i]%nv);
    }
  }
  return tr;
}

template <typename N_WORK_T>
std::vector<std::vector<N_WORK_T>> naive_solver(
    std::vector<std::vector<double>>& bw, std::vector<N_WORK_T>& load,
    double k) {
  int nv = load.size();
  for(int i=0; i<nv; ++i) bw[i][i] = k;
  std::vector<std::vector<N_WORK_T>> tr(nv);
  for(int i=0; i<nv; ++i){
    tr[i].resize(nv, 0);
    double work = load[i];
    double sum = 0;
    for(int j=0; j<nv; ++j) sum += bw[i][j];
    for(int j=0; j<nv; ++j) {
      if(i==j) continue;
       tr[i][j] = work / sum * bw[i][j];
    }
  }

  for(int i=0; i<nv; ++i){
    for(int j=i+1; j<nv; ++j){
      if(i==j) continue;
      if(tr[i][j] > tr[j][i]) { tr[i][j] -= tr[j][i]; tr[j][i] = 0;}
      else { tr[j][i] -= tr[i][j]; tr[i][j] = 0;}
    }
  }
  return tr;
}

/* @breif: Topology-aware workload distribution based
 *         on the min-cost-max-flow model.
 * Parameter:
 *  [in] bw[][]: the b/w between GPUi and GPUj
 *  [in] load[]: the amount of work-items in each GPU
 *  [in] k: the ratio of local efficiency / remote efficiency
 *  [out] tr[][]: the number of work-items should be sent to other GPUs
 * Example:
 * vector<vector<double>> bw = {{0,50,25,50,25,1,1,1},
 *                             {50,0,25,25,1,50,1,1},
 *                             {25,25,0,50,1,1,50,1},
 *                             {50,25,50,0,1,1,1,25},
 *                             {25,1,1,1,0,50,25,50},
 *                             {1,50,1,1,50,0,25,25},
 *                             {1,1,50,1,25,25,0,50},
 *                             {1,1,1,25,50,25,50,0}};
 * vector<int> load = {8, 1, 1, 1, 0, 0, 0, 0};
 * vector<vector<int>> tr(8,vector<int>(8));
 * k=1;
 * EK_solver(bw, load, k, tr);
 * 0 1 1 1 1 1 1 1
 * 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0
 * it means GPUi should send tr[i][j] work-item to GPUj
 */
template <typename N_WORK_T>
std::vector<std::vector<N_WORK_T>> EK_solver(
    std::vector<std::vector<double>>& bw, std::vector<N_WORK_T>& load,
    double k) {
  auto i_inf = std::numeric_limits<N_WORK_T>::max();
  std::vector<N_WORK_T> target;
  double h = 0;
  {
    auto probe = [&](double h) -> bool {
      double hi = 0, lo = 0;
      for (auto val : load) {
        if (val > h)
          hi += val - h;
        else if (val < h)
          lo += k * (h - val);
      }
      if (hi > lo)
        return true;  // increase h
      else
        return false;  // decrease h
    };

    N_WORK_T l = i_inf, r = 0;
    for (auto val : load) {
      l = std::min(val, l);
      r = std::max(val, r);
    }
    while (r - l > 1) {
      h = ((double) l + r) / 2.0;
      if (probe(h))
        l = h;
      else
        r = h;
    }
  }

  target.resize(load.size());
  for (int i = 0; i < load.size(); ++i) {
    if (load[i] > h)
      target[i] = h;
    else
      target[i] = (h - load[i]) / k + load[i];
  }

  // samll dense graph;
  typedef struct status_t {
    size_t cap{};
    double cost{};
    status_t(size_t a, double b) : cap(a), cost(b) {}
    status_t() = default;
  } status_t;

  int64_t sum = 0, target_sum = 0;
  for (auto val : load)
    sum += val;
  for (auto val : target)
    target_sum += val;
  int64_t diff = (target_sum - sum);
  int onv = load.size();

  if (diff < 0) {
    diff = -diff;
    while (diff) {
      auto flow = diff;
      int cnt = 0;
      for (int i = 0; i < onv; ++i) {
        if (load[i] > target[i]) {
          flow = std::min((int64_t) load[i] - (int64_t) target[i], flow);
          cnt++;
        }
      }
      auto change = std::min(diff, flow * cnt);
      auto p_change = change / cnt;
      auto remain = change % cnt;
      for (int i = 0; i < onv; ++i) {
        if (load[i] > target[i]) {
          target[i] += p_change;
          if (target[i] < load[i] && remain) {
            remain--;
            target[i]++;
          }
        }
      }
      diff = diff - change + remain;
    }
  } else {
    while (diff) {
      auto flow = diff;
      int cnt = 0;
      for (int i = 0; i < onv; ++i) {
        if (load[i] < target[i]) {
          flow = std::min((int64_t) target[i] - (int64_t) load[i], flow);
          cnt++;
        }
      }
      auto change = std::min(diff, flow * cnt);
      auto p_change = change / cnt;
      auto remain = change % cnt;
      for (int i = 0; i < onv; ++i) {
        if (load[i] < target[i]) {
          target[i] -= p_change;
          if (target[i] > load[i] && remain) {
            remain--;
            target[i]--;
          }
        }
      }
      diff = diff - change + remain;
    }
  }

  target_sum = 0;
  for (auto val : target)
    target_sum += val;
  assert(target_sum == sum);

  auto nv = onv + 2;
  auto s = onv;
  auto e = onv + 1;
  auto sup = sum + 1;
  double inf = i_inf;

  std::vector<std::vector<status_t>> g(nv, std::vector<status_t>(nv));

  for (int i = 0; i < onv; ++i) {
    if (load[i] > target[i])
      g[s][i].cap = load[i] - target[i];
  }

  for (int i = 0; i < onv; ++i) {
    if (load[i] < target[i])
      g[i][e].cap = target[i] - load[i];
  }

  for (int i = 0; i < onv; ++i) {
    for (int j = 0; j < onv; ++j) {
      if (load[i] > target[i] && load[j] < target[j]) {
        double cost = bw[i][j] == 0 ? inf : (1 / bw[i][j]);
        g[i][j].cap = sup;
        g[i][j].cost = cost;
        g[j][i].cap = 0;
        g[j][i].cost = -cost;
      }
    }
  }

  auto spfa = [&]() {
    std::vector<int> vis(nv, 0);
    std::vector<int> pre(nv, -1);
    std::vector<double> dis(nv, inf);
    std::queue<int> q;
    q.push(s);
    vis[s] = 1;
    dis[s] = 0;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      vis[u] = 0;
      for (int v = 0; v < nv; ++v) {
        if (g[u][v].cap == 0)
          continue;
        double cost = g[u][v].cost;
        if (dis[u] != inf && dis[v] - dis[u] - cost > 1e-7) {
          dis[v] = dis[u] + cost;
          pre[v] = u;
          if (vis[v] == 0) {
            vis[v] = 1;
            q.push(v);
          }
        }
      }
    }

    if (dis[e] == inf)
      return false;
    else {
      auto flow = i_inf;
      for (int c = e; c != s; c = pre[c]) {
        flow = std::min(flow, g[pre[c]][c].cap);
      }
      for (int c = e; c != s; c = pre[c]) {
        g[pre[c]][c].cap -= flow;
        g[c][pre[c]].cap += flow;
      }
      return true;
    }
  };

  while (spfa())
    ;

  std::vector<std::vector<N_WORK_T>> tr(onv);
  for (int i = 0; i < onv; ++i) {
    tr[i].resize(onv, 0);
    for (int j = 0; j < onv; ++j) {
      if (load[i] > target[i] && load[j] < target[j]) {
        tr[i][j] = sup - g[i][j].cap;
      }
    }
  }
  return tr;
}
}  // namespace grape_gpu

#endif
