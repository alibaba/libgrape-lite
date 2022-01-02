#!/usr/bin/env bash
source $(dirname "$0")/eval_base.sh

function run_app() {
  mtx=$1
  app=$2
  source=$3
  device=$4

  # Run 5 times at most, max number of iterations for pagerank is 10
  echo "Evaluating gunrock - $app with $mtx"
  for i in {1..5}; do
    cmd_output=$("${gunrock_prefix}"/"${app}" market "$mtx" --device="$device" --quick --iteration-num=5 --max-iter=10)
    if [[ $? -eq 0 ]]; then
      break
    fi
  done

  runtime=$(echo "$cmd_output" | perl -nle 'print "$1" if /avg. elapsed: (.*?) ms/')
  if [[ -z "$runtime" || "$runtime" == "0.0000" ]]; then
    runtime="failed"
  fi
  device=${device//,/ }
  echo "$(date),$app,$mtx,$source,$device,$runtime" >>"gunrock_${app}.csv"
}

if [ "$#" -ne 2 ]; then
  echo "Usage: eval_gunrock.sh [app name] [device list]" >&2
  exit 1
fi

app=$1
device=$2

echo "date,app,dataset,source,device,time" >>"gunrock_${app}.csv"

for ((idx = 0; idx < ${#dataset_source_node[@]}; idx += 2)); do
  dataset=${dataset_source_node[idx]}
  source=${dataset_source_node[idx + 1]}
  dataset=${dataset_prefix}/${dataset}/${dataset}.ud.random.weight.mtx

  run_app "$dataset" "$app" "$source" "$device"
done
