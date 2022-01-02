#!/usr/bin/env bash

dataset_base="/root/data/liang/new_dataset"

dataset=(
  "$dataset_base/soc-LiveJournal1/soc-LiveJournal1.mtx"
  "$dataset_base/hollywood-2009/hollywood-2009.mtx"
  "$dataset_base/soc-orkut/soc-orkut.mtx"
  "$dataset_base/soc-sinaweibo/soc-sinaweibo.mtx"
  "$dataset_base/soc-twitter-2010/soc-twitter-2010.mtx"
  "$dataset_base/indochina-2004/indochina-2004.mtx"
  "$dataset_base/uk-2002/uk-2002.mtx"
  "$dataset_base/arabic-2005/arabic-2005.mtx"
  "$dataset_base/uk-2005/uk-2005.mtx"
  "$dataset_base/webbase-2001/webbase-2001.mtx"
)

dataset_weight=(
  "$dataset_base/soc-LiveJournal1/soc-LiveJournal1.random.weight.mtx"
  "$dataset_base/hollywood-2009/hollywood-2009.random.weight.mtx"
  "$dataset_base/soc-orkut/soc-orkut.random.weight.mtx"
  "$dataset_base/soc-sinaweibo/soc-sinaweibo.random.weight.mtx"
  "$dataset_base/soc-twitter-2010/soc-twitter-2010.mtx"
  "$dataset_base/indochina-2004/indochina-2004.random.weight.mtx"
  "$dataset_base/uk-2002/uk-2002.random.weight.mtx"
  "$dataset_base/arabic-2005/arabic-2005.random.weight.mtx"
  "$dataset_base/uk-2005/uk-2005.random.weight.mtx"
  "$dataset_base/webbase-2001/webbase-2001.random.weight.mtx"
)

names=(
  "livejournal_86M"
  "hollywood_113M"
  "orkut_213M"
  "sinaweibo_523M"
  "twitter_530M"
  "indochina_302M"
  "uk-2002_524M"
  "arabic_1.11B"
  "uk-2005_1.57B"
  "webbase_1.71B"
)

app=$1

function get_device() {
  n_dev=$1
  dev_list="0"
  for ((i = 1; i < $n_dev; i++)); do
    dev_list="$dev_list,$i"
  done
  echo $dev_list
}

for n_gpu in 1 2 4 8; do
  dev_list=$(get_device 8)
  for j in "${!names[@]}"; do
    if [[ $app == "sssp" ]]; then
      dataset_path=${dataset_weight[j]}
    else
      dataset_path=${dataset[j]}
    fi
    log_path=/root/data/liang/gunrock_expr/json/${app}_${n_gpu}_${names[i]}.json
    if [[ ! -f $log_path ]]; then
      /root/data/liang/gunrock/build/bin/$app market $dataset_path --src=1 --device=$dev_list --quick --jsonfile=$log_path
    fi
  done
done
