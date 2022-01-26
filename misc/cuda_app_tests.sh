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

  cmd="mpirun -n ${NP} ./run_cuda_app --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --application ${APP} --out_prefix ./extra_tests_output $@"
  echo ${cmd}
  eval ${cmd}
}

function RunWeightedApp() {
  NP=$1; shift
  APP=$1; shift

  cmd="mpirun -n ${NP} ./run_cuda_app --vfile ${GRAPE_HOME}/dataset/${GRAPH}.v --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --application ${APP} --out_prefix ./extra_tests_output $@"
  echo ${cmd}
  eval ${cmd}
}

function RunAppWithELoader() {
  NP=$1; shift
  APP=$1; shift

  cmd="mpirun -n ${NP} ./run_cuda_app --efile ${GRAPE_HOME}/dataset/${GRAPH}.e --application ${APP} --out_prefix ./extra_tests_output --nosegmented_partition $@"
  echo ${cmd}
  eval ${cmd}
}

pushd ${GRAPE_HOME}/build

g++ ${GRAPE_HOME}/misc/wcc_check.cc -std=c++11 -O3 -o ./wcc_check
g++ ${GRAPE_HOME}/misc/eps_check.cc -std=c++11 -O3 -o ./eps_check

nproc=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ ${nproc} -gt 8 ]; then
  nproc=8
fi
proc_list="1 $(seq 2 6 ${nproc})"
lb_list="none wm cm cta strict"

for np in ${proc_list}; do
  for lb in ${lb_list}; do
    RunWeightedApp ${np} sssp -lb=${lb} --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

    RunWeightedApp ${np} sssp -lb=${lb} --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP-directed

    RunAppWithELoader ${np} sssp -lb=${lb} --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP

    RunAppWithELoader ${np} sssp -lb=${lb} --sssp_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-SSSP-directed

    RunApp ${np} bfs -lb=${lb} --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

    RunApp ${np} bfs -lb=${lb} --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS-directed

    RunAppWithELoader ${np} bfs -lb=${lb} --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH}
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS

    RunAppWithELoader ${np} bfs -lb=${lb} --bfs_source=6 --serialize=true --serialization_prefix=./serial/${GRAPH} --directed
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-BFS-directed

    RunApp ${np} pagerank -lb=${lb} --pr_mr=10 --pr_d=0.85
    EpsVerify ${GRAPE_HOME}/dataset/${GRAPH}-PR

    RunApp ${np} cdlp -lb=${lb} --cdlp_mr=10
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-CDLP

    RunApp ${np} lcc -lb=${lb} --serialize=true --serialization_prefix=./serial/${GRAPH}
    ExactVerify ${GRAPE_HOME}/dataset/${GRAPH}-LCC

    RunApp ${np} wcc -lb=${lb}
    WCCVerify ${GRAPE_HOME}/dataset/${GRAPH}-WCC
  done
done

popd
