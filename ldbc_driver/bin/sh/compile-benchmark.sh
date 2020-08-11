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


# Ensure the configuration file exists
rootdir="$( cd "$(dirname "$0")/../.." >/dev/null 2>&1 ; pwd -P )"

config="${rootdir}/config/"

if [ ! -f "$config/platform.properties" ]; then
	echo "Missing mandatory configuration file: $config/platform.properties" >&2
	exit 1
fi


# Construct the classpath
LIBGRAPE_HOME=$(grep -E "^platform.libgrape.home[	 ]*[:=]" $config/platform.properties | sed 's/platform.libgrape.home[	 ]*[:=][	 ]*\([^	 ]*\).*/\1/g' | head -n 1)
if [ -z LIBGRAPE_HOME ]; then
    echo "Error: home directory for libgrape not specified."
    echo "Define the environment variable \$LIBGRAPE_HOME or modify platform.libgrape.home in $config/platform.properties"
    exit 1
fi
GRANULA_ENABLED=$(grep -E "^benchmark.run.granula.enabled[	 ]*[:=]" $config/granula.properties | sed 's/benchmark.run.granula.enabled[	 ]*[:=][	 ]*\([^	 ]*\).*/\1/g' | head -n 1)


# Build binaries
mkdir -p bin/standard
(cd bin/standard && cmake -DCMAKE_BUILD_TYPE=Release ${LIBGRAPE_HOME} && make analytical_apps)

if [ "$GRANULA_ENABLED" = "true" ] ; then
 mkdir -p bin/granula
 (cd bin/granula && cmake -DCMAKE_BUILD_TYPE=Release -DGRANULA=1 ${LIBGRAPE_HOME} && make analytical_apps)
fi

if [ $? -ne 0 ]
then
    echo "compilation failed"
    exit 1
fi
