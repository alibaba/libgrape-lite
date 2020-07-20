# Graphalytics libgrape-lite driver


### Getting started

This is a [Graphalytics](https://github.com/ldbc/ldbc_graphalytics/) benchmark driver for **libgrape-lite**. This driver is derived from [the driver for PowerGraph](https://github.com/atlarge-research/graphalytics-platforms-powergraph), implemented by [atlarge-research](https://github.com/atlarge-research). In addition, please refer to the documentation of [Graphalytics core](https://github.com/ldbc/ldbc_graphalytics) for an introduction to using Graphalytics. The basic usage is:

  - Download the source code.
  - Execute `mvn clean package` in the this directory (See details in [Software Build](https://github.com/ldbc/ldbc_graphalytics/wiki/Documentation:-Software-Build)).
  - Extract the distribution from  `graphalytics-{graphalytics-version}-grape-{platform-version}.tar.gz`.
  - Check configuration files under `config` to ensure everything is correct.
  - run `./bin/sh/run-benchmark.sh`.

### Specific configuration for libgrape-lite

To run benchmark with the driver, some configurations need to be assigned according to your environment. Edit `config/grape.properties` to change the following settings:

 - `platform.grape.home`: Set to the root directory of the **libgrape-lite**.
 - `platform.grape.nodes`: Set the the names of computation nodes, with format e.g., `10.149.0.55\,10.149.0.56` (note: IPs are separated by `\,` instead of spaces).

### Running the benchmark

To execute a Graphalytics benchmark on **libgrape-lite** (using this driver), follow the steps in the [Graphalytics tutorial](https://github.com/ldbc/ldbc_graphalytics/wiki/Manual%3A-Running-Benchmark).

Alternatively, you can execute `run_ldbc.sh' to run the benchmark in an automatic manner. The script will demo the process with a moderate sized dataset on your local machine. Feel free to change the configurations in the script to run on your own dataset or in a distributed environment.
