#!/bin/bash
#
# Copyright 2015 Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

for host in `echo $1 | tr ',' ' '`;
do
  ssh $host mkdir -p `dirname $5`
  scp $5 $host:$5
done

LOG_PATH=$2
LIBGRAPE_HOME=$3
FINAL_OUT=$4
echo ${@:5}
# the binary is sync to ${HOME}/bin/standard/run_app
# switch to $HOME before mpirun
pushd ${HOME}
mpirun --map-by ppr:1:node --bind-to none --host $1 ${@:5} &
popd

echo $! > $LOG_PATH/executable.pid
wait $!
