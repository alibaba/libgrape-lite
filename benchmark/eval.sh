#!/usr/bin/env bash

eval_home=$(dirname "$0")

for i in 1 4 8; do
  "$eval_home"/eval_gum.sh pagerank $i
  "$eval_home"/eval_gum.sh pagerank_pull $i
  "$eval_home"/eval_gum.sh bfs $i
  "$eval_home"/eval_gum.sh sssp $i
  "$eval_home"/eval_gum.sh wcc $i
  "$eval_home"/eval_gum.sh wcc_opt $i
done

"$eval_home"/eval_gunrock.sh bfs 0
"$eval_home"/eval_gunrock.sh sssp 0
"$eval_home"/eval_gunrock.sh pr 0
"$eval_home"/eval_gunrock.sh cc 0



