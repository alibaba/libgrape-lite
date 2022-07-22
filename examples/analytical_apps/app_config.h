
#ifndef EXAMPLES_ANALYTICAL_APPS_APP_CONFIG_H_
#define EXAMPLES_ANALYTICAL_APPS_APP_CONFIG_H_
#include "grape_gpu/parallel/parallel_engine.h"
namespace grape_gpu {
struct AppConfig {
  float wl_alloc_factor_in;
  float wl_alloc_factor_out_local;
  float wl_alloc_factor_out_remote;
  LoadBalancing lb;
  bool work_stealing;
  double ws_k;
};
}  // namespace grape_gpu
#endif  // EXAMPLES_ANALYTICAL_APPS_APP_CONFIG_H_
