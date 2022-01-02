#!/usr/bin/env bash
source $(dirname "$0")/eval_base.sh

function run_app() {
  gr=$1
  app=$2
  source=$3
  num_gpus=$4

  cmd="${groute_prefix}/${app} -graphfile $gr -startwith=$num_gpus -num_gpus=$num_gpus"

  if [[ $app == "bfs" || $app == "sssp" ]]; then
    cmd="$cmd -source_node $source"
  elif [[ $app == "cc" ]]; then
    is_directed="-undirected=false"
    if [[ "$gr" == *.ud.* ]]; then
      is_directed=""
    fi
    cmd="$cmd $is_directed"
  elif [[ $app == "pr" ]]; then
    cmd="$cmd -noopt -wl_alloc_factor_local=0.1 -wl_alloc_factor_in=0.1 -wl_alloc_factor_out=0.3 -wl_alloc_factor_pass=0.5"
  fi
  echo "Evaluating groute - $app with $gr"
  runtime=$($cmd | grep "filter" | awk '{print $(NF-2)}')
  if [[ $? -ne 0 || -z "$runtime" ]]; then
    runtime="failed"
  fi

  echo "$(date),$app,$gr,$source,$num_gpus,$runtime" >>"groute_${app}.csv"
}

if [ "$#" -ne 2 ]; then
  echo "Usage: eval_groute.sh [app name] [num of gpus]" >&2
  exit 1
fi

app=$1
num_gpus=$2

echo "date,app,dataset,source,num_gpus,time" >>"groute_${app}.csv"

for ((idx = 0; idx < ${#dataset_source_node[@]}; idx += 2)); do
  dataset=${dataset_source_node[idx]}
  source=${dataset_source_node[idx + 1]}
  dataset=${dataset_prefix}/${dataset}/${dataset}.ud.random.weight.gr

  run_app "$dataset" "$app" "$source" "$num_gpus"
done
