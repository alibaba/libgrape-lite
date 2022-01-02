#!/usr/bin/env bash
source $(dirname "$0")/eval_base.sh

function run_app() {
  efile=$1
  app=$2
  lb=$3
  np=$4
  partitioner=$5
  prio=$6
  source=$7
  echo "Evaluating grape - $app with $efile"
  OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np $np ${grape_prefix}/run_app \
    -efile ${efile} \
    -mtx \
    -application=$app \
    -rebalance=false \
    -partitioner=$partitioner \
    -directed=false \
    -rm_self_cycle=true \
    -serialization_prefix=${grape_ser_prefix} \
    -lb=$lb \
    -bfs_source=$source \
    -sssp_source=$source \
    -sssp_prio=$prio 2>&1 | tee grape-gpu.running.log
  runtime=$(perl -nle 'print "$1" if /run algorithm: (.*?) sec/' <grape-gpu.running.log)

  echo "$(date),$app,$efile,$lb,$partitioner,$np,$runtime" >>"grape_${app}".csv
}

sssp_prio=64

# sssp pagerank bfs
app=$1
num_gpus=$2

echo "date,app,efile,lb,partitioner,np,time" >>grape_${app}.csv

for ((idx = 0; idx < ${#dataset_source_node[@]}; idx += 2)); do
  dataset=${dataset_source_node[idx]}
  source=${dataset_source_node[idx + 1]}
  dataset=${dataset_prefix}/${dataset}/${dataset}.ud.random.weight.mtx

  for lb in cta wm cm cmold strict; do
    # segmented partitioner
    run_app $dataset $app $lb $num_gpus seg $sssp_prio $source
    # hash partitioner
    run_app $dataset $app $lb $num_gpus hash $sssp_prio $source
  done
done
