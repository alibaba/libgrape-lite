#!/bin/bash -e
GRAPE_HOME="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"

GRAPH=p2p-31

function ExactVerify() {
  cat ./extra_tests_output/* | sort -k1n > ./extra_tests_tmp.res
  if ! cmp ./extra_tests_tmp.res $1 > /dev/null 2>&1
  then
    echo "Wrong answer"
    exit 1
  else
    rm -rf ./extra_tests_output/*
    rm -rf ./extra_tests_tmp.res
  fi
}

function EpsVerify() {
  cat ./extra_tests_output/* | sort -k1n > ./extra_tests_tmp.res
  if ! ./eps_check ./extra_tests_tmp.res $1 > /dev/null 2>&1
  then
    echo "Wrong answer"
    exit 1
  else
    rm -rf ./extra_tests_output/*
    rm -rf ./extra_tests_tmp.res
  fi
}

function WCCVerify() {
  cat ./extra_tests_output/* | sort -k1n > ./extra_tests_tmp.res
  if ! ./wcc_check ./extra_tests_tmp.res $1 > /dev/null 2>&1
  then
    echo "Wrong answer"
    exit 1
  else
    rm -rf ./extra_tests_output/*
    rm -rf ./extra_tests_tmp.res
  fi
}

function RunApp() {
  NP=$1; shift
  APP=$1; shift

  cmd="mpirun -n ${NP} ./run_app --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --application ${APP} --out_prefix ./extra_tests_output $@"
  echo ${cmd}
  eval ${cmd}
}

function BasicTests() {
  np=$1; shift

  RunApp ${np} sssp --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  RunApp ${np} sssp_auto --sssp_source=6 --deserialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  RunApp ${np} sssp --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP-directed

  RunApp ${np} sssp_auto --sssp_source=6 --deserialize=true --serialization_prefix=./serial/${GRAPH} --directed
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP-directed

  RunApp ${np} bfs --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

  RunApp ${np} bfs_auto --bfs_source=6 --deserialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

  RunApp ${np} bfs --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS-directed

  RunApp ${np} bfs_auto --bfs_source=6 --deserialize=true --serialization_prefix=./serial/${GRAPH} --directed
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS-directed

  RunApp ${np} pagerank --pr_mr=10 --pr_d=0.85
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

  RunApp ${np} pagerank_auto --pr_mr=10 --pr_d=0.85
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

  RunApp ${np} pagerank_local --pr_mr=10 --pr_d=0.85
  RunApp ${np} pagerank_local_parallel --pr_mr=10 --pr_d=0.85

  RunApp ${np} pagerank_parallel --pr_mr=10 --pr_d=0.85
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

  RunApp ${np} pagerank_parallel --pr_mr=10 --pr_d=0.85 --directed
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR-directed

  RunApp ${np} pagerank_auto --pr_mr=10 --pr_d=0.85 --directed
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR-directed

  RunApp ${np} cdlp --cdlp_mr=10
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-CDLP

  RunApp ${np} cdlp_auto --cdlp_mr=10
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-CDLP

  RunApp ${np} lcc --serialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-LCC

  RunApp ${np} lcc_auto --deserialize=true --serialization_prefix=./serial/${GRAPH}
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-LCC

  RunApp ${np} wcc
  WCCVerify ${GRAPE_HOME}/dataset/${GRAPH}-WCC

  RunApp ${np} wcc_auto
  WCCVerify ${GRAPE_HOME}/dataset/${GRAPH}-WCC
}

function MutableFragmentTest() {
  NP=$1; shift
  APP=$1; shift

  cmd="mpirun -n ${NP} ./mutable_fragment_tests --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e.mutable_base --delta_efile ${GRAPE_HOME}/dataset/${GRAPH}.e.mutable_delta --application ${APP} --out_prefix ./extra_tests_output $@"
  echo ${cmd}
  eval ${cmd}
}

function MutableFragmentTests() {
  np=$1; shift

  MutableFragmentTest ${np} sssp --sssp_source=6
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  MutableFragmentTest ${np} sssp_auto --sssp_source=6
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  MutableFragmentTest ${np} bfs --bfs_source=6
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

  MutableFragmentTest ${np} bfs_auto --bfs_source=6
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

  MutableFragmentTest ${np} pagerank_auto --pr_mr=10 --pr_d=0.85
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

  MutableFragmentTest ${np} pagerank_local_parallel --pr_mr=10 --pr_d=0.85

  MutableFragmentTest ${np} pagerank_parallel --pr_mr=10 --pr_d=0.85
  EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

  MutableFragmentTest ${np} cdlp --cdlp_mr=10
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-CDLP

  MutableFragmentTest ${np} cdlp_auto --cdlp_mr=10
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-CDLP

  MutableFragmentTest ${np} lcc
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-LCC

  MutableFragmentTest ${np} lcc_auto
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-LCC

  MutableFragmentTest ${np} wcc
  WCCVerify ${GRAPE_HOME}/dataset/${GRAPH}-WCC

  MutableFragmentTest ${np} wcc_auto
  WCCVerify ${GRAPE_HOME}/dataset/${GRAPH}-WCC
}

function VertexMapTest() {
  NP=$1; shift

  cmd="mpirun -n ${NP} ./vertex_map_tests --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --out_prefix ./extra_tests_output --sssp_source=6 $@"

  echo ${cmd}
  eval ${cmd}
}

function VertexMapTestOnMutableFragment() {
  NP=$1; shift

  cmd="mpirun -n ${NP} ./vertex_map_tests --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e.mutable_base --delta_efile ${GRAPE_HOME}/dataset/${GRAPH}.e.mutable_delta --out_prefix ./extra_tests_output --sssp_source=6 $@"

  echo ${cmd}
  eval ${cmd}
}

function VertexMapTests() {
  np=$1; shift

  VertexMapTest ${np} --string_id
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --nosegmented_partition
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --string_id --nosegmented_partition
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --noglobal_vertex_map
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --string_id --noglobal_vertex_map
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --nosegmented_partition --noglobal_vertex_map
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTest ${np} --string_id --nosegmented_partition --noglobal_vertex_map
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTestOnMutableFragment ${np} --string_id
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTestOnMutableFragment ${np} --nosegmented_partition
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

  VertexMapTestOnMutableFragment ${np} --string_id --nosegmented_partition
  ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP
}

pushd ${GRAPE_HOME}/build

g++ ${GRAPE_HOME}/misc/wcc_check.cc -std=c++11 -O3 -o ./wcc_check
g++ ${GRAPE_HOME}/misc/eps_check.cc -std=c++11 -O3 -o ./eps_check

nproc=$(getconf _NPROCESSORS_ONLN)
if [ ${nproc} -gt 8 ]; then
  nproc=8
fi
proc_list="1 $(seq 2 2 ${nproc})"

for np in ${proc_list}; do
  BasicTests ${np}
  MutableFragmentTests ${np}
  VertexMapTests ${np}
done

popd
