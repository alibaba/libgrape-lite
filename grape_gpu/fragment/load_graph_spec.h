#ifndef GRAPE_GPU_FRAGMENT_LOAD_GRAPH_SPEC_H_
#define GRAPE_GPU_FRAGMENT_LOAD_GRAPH_SPEC_H_
#include <string>

namespace grape_gpu {

/**
 * @brief LoadGraphSpec determines the specification to load a graph.
 *
 */
struct LoadGraphSpec {
  bool directed;
  bool skip_first_valid_line;
  bool rebalance;
  int rebalance_vertex_factor;
  bool rm_self_cycle;

  bool serialize;
  std::string serialization_prefix;

  bool deserialize;
  std::string deserialization_prefix;

  int kronec_scale;
  int kronec_edgefactor;

  void set_directed(bool val = true) { directed = val; }

  void set_skip_first_valid_line(bool val = false) {
    skip_first_valid_line = val;
  }

  void set_rebalance(bool flag, int weight) {
    rebalance = flag;
    rebalance_vertex_factor = weight;
  }

  void set_rm_self_cycle(bool val = true) { rm_self_cycle = val; }

  void set_serialize(bool flag, const std::string& prefix) {
    serialize = flag;
    serialization_prefix = prefix;
  }

  void set_deserialize(bool flag, const std::string& prefix) {
    deserialize = flag;
    deserialization_prefix = prefix;
  }

  void set_kronec(int _scale, int _edgefactor) {
    kronec_scale = _scale;
    kronec_edgefactor = _edgefactor;

  }
};

inline LoadGraphSpec DefaultLoadGraphSpec() {
  LoadGraphSpec spec;
  spec.directed = true;
  spec.skip_first_valid_line = false;
  spec.rebalance = true;
  spec.rebalance_vertex_factor = 0;
  spec.rm_self_cycle = true;
  spec.serialize = false;
  spec.deserialize = false;
  return spec;
}
}  // namespace grape_gpu

#endif  // GRAPE_GPU_FRAGMENT_LOAD_GRAPH_SPEC_H_
