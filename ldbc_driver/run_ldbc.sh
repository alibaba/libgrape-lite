#!/bin/bash
set -ex

# normal or ci mode
MODE="normal"

# we use a moderate size graph as example, see full list of dataset at https://graphalytics.org/datasets
GRAPH_NAME="datagen-7_6-fb"

# if in CI mode, use a smaller graph
if [[ $1 == "ci" ]]; then
    MODE="ci"
    GRAPH_NAME="p2p-31"
fi

# environment variables, change them if needed.
LIBGRAPE_HOME="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"
LDBC_HOME="${LIBGRAPE_HOME}/ldbc_driver"
WORKSPACE="${LDBC_HOME}/workspace"
LOG_FILE="${WORKSPACE}/output.log"

# MPI nodes, in the format of "10.149.0.55\,10.149.0.56".
# extra slashes here is for escape
HOST_NODES="127.0.0.1\\\,127.0.0.1"

# two random ports for executor and runner port.
EXECUTOR_PORT="$(awk 'BEGIN{srand();print int(rand()*(30000-2000))+10000 }')"
RUNNER_PORT="$(awk 'BEGIN{srand();print int(rand()*(50000-30001))+10000 }')"

# check the existance of libgrape_lite.so, build if not exists.
if [[ ! -f "${LIBGRAPE_HOME}/build/libgrape_lite.so" ]]; then
    mkdir -p "${LIBGRAPE_HOME}/build"
    pushd "${LIBGRAPE_HOME}/build"
    cmake ..
    make
    popd
fi

# check the existance of the tar of driver, build with maven if not exists.
pushd ${LDBC_HOME}
if [[ ! -f "graphalytics-1.0.0-libgrape-0.3-SNAPSHOT-bin.tar.gz" ]]; then
    mvn clean package -DskipTests
fi

# extract the driver to the workspace if not exists.
if [[ ! -d "${WORKSPACE}/graphalytics-1.0.0-libgrape-0.3-SNAPSHOT" ]]; then
    mkdir -p ${WORKSPACE}
    tar xzf graphalytics-1.0.0-libgrape-0.3-SNAPSHOT-bin.tar.gz -C ${WORKSPACE} # TODO: version 0.3
fi

pushd ${WORKSPACE}

# download data.
if [[ ! -d "graphs" ]]; then
    mkdir ./graphs
    if [[ ${MODE} == "normal" ]]; then
        # mirrored from https://graphalytics.org/datasets to speed up.
        wget https://libgrape-lite.oss-cn-zhangjiakou.aliyuncs.com/${GRAPH_NAME}.zip
        unzip -j ${GRAPH_NAME}.zip -d ./graphs
    elif [[ ${MODE} == "ci" ]]; then
        cp ${LIBGRAPE_HOME}/dataset/* ./graphs
    fi
fi

pushd graphalytics-1.0.0-libgrape-0.3-SNAPSHOT

# config the properties for ldbc-driver.
if [[ ! -d "config" ]]; then
    cp -r ./config-template ./config

    # use '#' rather than '/' to avoid potential '/' in ${LIBGRAPE_HOME}
    sed -i'.bak' '/^platform.libgrape.home/ s#$# '"${LIBGRAPE_HOME}"'#' config/platform.properties
    sed -i'.bak' '/^platform.libgrape.nodes/ s/$/ '"${HOST_NODES}"'/' config/platform.properties
    sed -i'.bak' -e '/^benchmark.executor.port/ s/$/ '"${EXECUTOR_PORT}"'/' -e '/^benchmark.runner.port/ s/$/ '"${RUNNER_PORT}"'/' config/benchmark.properties
    sed -i'.bak' '/^graphs.root-directory/ s#$# '"${WORKSPACE}/graphs"'#' config/benchmark.properties
    sed -i'.bak' '/^graphs.validation-directory/ s#$# '"${WORKSPACE}/graphs"'#' config/benchmark.properties
    sed -i'.bak' '/^graphs.output-directory/ s#$# '"${WORKSPACE}/output"'#' config/benchmark.properties
    sed -i'.bak' '/^benchmark.custom.graphs / s/$/ '"${GRAPH_NAME}"'/' config/benchmarks/custom.properties
fi

# clean up the binaries
rm -rf ./bin/standard || true

# run ldbc benchmarking suite.
./bin/sh/run-benchmark.sh | tee ${LOG_FILE}

grep "succeed." ${LOG_FILE} | grep "6 / 6"

if [[ $? -eq 0 ]]; then
    echo "Finished successfully."
else
    echo "failed to run ldbc-benchmark."
    echo "detailed log: ${LOG_FILE}."
    exit 1
fi
