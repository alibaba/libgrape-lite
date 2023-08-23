#!/bin/bash
export DATA_PATH=/mnt/data/nfs/dataset
export GB_PATH=/mnt/data/nfs/graphbolt
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset=('road_usa' 'europe_osm' 'uk-2005' 'twitter')
dataset=('uk-2005')
ext=('base' 'update' 'updated')
perc=$1

for name in "${dataset[@]}"
do
  echo "Generating $name"
  if [[ $name == "twitter" ]]; then
    "$DIR"/gen.py $DATA_PATH/"$name"/"$name".edges $perc
  else
    "$DIR"/gen.py $DATA_PATH/"$name"/"$name".mtx $perc -header
  fi

  echo "Cutting to generate non-weighted graph"
  for ext in "${ext[@]}"
  do
    if [[ $ext == "update" ]]; then
      cut -d" " -f1,2,3 $DATA_PATH/"$name"/$perc/"$name"_w."$ext" > $DATA_PATH/"$name"/$perc/"$name"."$ext"
    else
      cut -d" " -f1,2 $DATA_PATH/"$name"/$perc/"$name"_w."$ext" > $DATA_PATH/"$name"/$perc/"$name"."$ext"
    fi
  done

  echo "Generating undirected graph"
  "$DIR"/d2ud.py $DATA_PATH/"$name"/$perc/"$name".base
  "$DIR"/d2ud.py $DATA_PATH/"$name"/$perc/"$name".update
  "$DIR"/d2ud.py $DATA_PATH/"$name"/$perc/"$name".updated

  echo "Converting to adj"
  $GB_PATH/tools/converters/SNAPtoAdjConverter $DATA_PATH/"$name"/$perc/"$name".base $DATA_PATH/"$name"/$perc/"$name".adj
  $GB_PATH/tools/converters/SNAPtoAdjConverter -w $DATA_PATH/"$name"/$perc/"$name"_w.base $DATA_PATH/"$name"/$perc/"$name"_w.adj
  $GB_PATH/tools/converters/SNAPtoAdjConverter $DATA_PATH/"$name"/$perc/"$name"_ud.base $DATA_PATH/"$name"/$perc/"$name"_ud.adj
done