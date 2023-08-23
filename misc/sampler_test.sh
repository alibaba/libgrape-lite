#!/bin/bash -e
GRAPE_HOME="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"

GRAPH=p2p-31

function RunSamplerOnStaticGraph {
  cmd_random="mpirun -n 4 ./run_sampler --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --sampling_strategy random --hop_and_num 4-5 --out_prefix ./output_sampling"
  echo "${cmd_random}"
  eval ${cmd_random}
  cmd_weight="mpirun -n 4 ./run_sampler --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --sampling_strategy edge_weight --hop_and_num 4-5 --out_prefix ./output_sampling"
  echo "${cmd_weight}"
  eval ${cmd_weight}
  cmd_top_k="mpirun -n 4 ./run_sampler --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --sampling_strategy top_k --hop_and_num 4-5 --out_prefix ./output_sampling"
  echo "${cmd_top_k}"
  eval ${cmd_top_k}
}

pushd ${GRAPE_HOME}/build
RunSamplerOnStaticGraph
popd
