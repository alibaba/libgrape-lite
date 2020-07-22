<h1 align="center">
    <img src="https://alibaba.github.io/libgrape-lite/logo.png" width="100" alt="libgrape-lite">
    <br>
    libgrape-lite
</h1>
<p align="center">
    A C++ library for parallel graph processing
</p>

[![C/C++ CI](https://github.com/alibaba/libgrape-lite/workflows/C++%20CI/badge.svg)](https://github.com/alibaba/libgrape-lite/actions?workflow=C++%20CI)
[![codecov](https://codecov.io/gh/alibaba/libgrape-lite/branch/master/graph/badge.svg)](https://codecov.io/gh/alibaba/libgrape-lite)

**libgrape-lite** is a C++ library from Alibaba for parallel graph processing. It differs from prior systems in its ability to parallelize sequential graph algorithms as a whole by following the *PIE* programming model from [GRAPE](https://dl.acm.org/doi/10.1145/3035918.3035942). Sequential algorithms can be easily ["plugged into"](examples/analytical_apps/sssp/sssp_auto.h) libgrape-lite with only minor changes and get parallelized to handle large graphs efficiently. In addition to the ease of programming, libgrape-lite is designed to be highly [efficient](Performance.md) and [flexible](examples/gnn_sampler), to cope the scale, variety and complexity from real-life graph applications.

## Building **libgrape-lite**

### Dependencies
**libgrape-lite** is developed and tested on CentOS 7. It should also work on other unix-like distributions. Building libgrape-lite requires the following softwares installed as dependencies.

- [CMake](https://cmake.org/) (>=2.8)
- A modern C++ compiler compliant with C++-11 standard. (g++ >= 4.8.1 or clang++ >= 3.3)
- [MPICH](https://www.mpich.org/) (>= 2.1.4) or [OpenMPI](https://www.open-mpi.org/) (>= 3.0.0)
- [glog](https://github.com/google/glog) (>= 0.3.4)


Here are the dependencies for optional features:
- [jemalloc](http://jemalloc.net/) (>= 5.0.0) for better memory allocation;
- [Doxygen](https://www.doxygen.nl/index.html) (>= 1.8) for generating documentation;
- Linux [HUGE_PAGES](http://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt) support, for better performance.

Extra dependencies are required by examples:
- [gflags](https://github.com/gflags/gflags) (>= 2.2.0);
- [boost](https://www.boost.org/) (>= 1.58);
- [Apache Kafka](https://github.com/apache/kafka) (>= 2.3.0);
- [librdkafka](https://github.com/edenhill/librdkafka)(>1.4.0) as the dependency for [cppkafka](https://github.com/mfontanini/cppkafka) (included as a submodule).


### Building libgrape-lite and examples

Once the required dependencies have been installed, go to the root directory of libgrape-lite and do a out-of-source build using CMake.

```bash
mkdir build && cd build
cmake ..
make -j
```

The building targets include a shared/static library, and two sets of examples: [analytical_apps](./examples/analytical_apps) and a [gnn_sampler](./examples/gnn_sampler).

Alternatively, you can build a particular target with command:

```bash
make libgrape-lite # or
make analytical_apps # or
make gnn_sampler
```

## Running libgrape-lite applications

### Graph format

The input of libgrape-lite is formatted following the [LDBC Graph Analytics](http://graphalytics.org) benchmark, with two files for each graph, a `.v` file for vertices with 1 or 2 columns, which are a vertex_id and optionally followed by the data assigned to the vertex; and a `.e` file for edges with 2 or 3 columns, representing source, destination and optionally the data on the edge, correspondingly. See sample files `p2p-31.v` and `p2p-31.e` under the [dataset](dataset/) directory. 

### Example applications

**libgrape-lite** provides six algorithms from the LDBC benchmark as examples. The deterministic algorithms are, single-source shortest path(SSSP), connected component(WCC), PageRank, local clustering coefficient(LCC), community detection of label propagation(CDLP), and breadth first search(BFS).    

To run a specific analytical application, users may use command like this:

```bash
# run single-source shortest path with 4 workers in local.
mpirun -n 4 ./run_app --vfile ../dataset/p2p-31.v --efile ../dataset/p2p-31.e --application sssp --sssp_source 0 --out_prefix ./output_sssp --directed

# or run connected component with 4 workers on a cluster.
# HOSTFILE provides a list of hosts where MPI processes are launched. 
mpirun -n 4 -hostfile HOSTFILE ./run_app --application=wcc --vfile ../dataset/p2p-31.v --efile ../dataset/p2p-31.e --out_prefix ./output_wcc

# see more flags info.
./run_app --help
```

### LDBC benchmarking

The analytical applications support the LDBC Analytical Benchmark suite with the provided `ldbc_driver`. Please refer to [ldbc_driver](./ldbc_driver) for more details. The benchmark results for libgrape-lite and other state-of-the-art systems could be found [here](Performance.md).

### GNN sampler

In addition to offline graph analytics, libgrape-lite could also be utilized to handle more complex graph tasks. A sampler for GNN training/inference on dynamic graphs (taking graph changes and queries, and producing results via [Kafka](https://kafka.apache.org/)) is included as an example. Please refer to [examples/gnn_sampler](./examples/gnn_sampler) for more details.

## Documentation

Documentation is generated using Doxygen. Users can build doxygen documentation in the build directory using:

```bash
cd build
make doc
# open docs/index.html
```

The latest version of online documentation can be found at [https://alibaba.github.io/libgrape-lite](https://alibaba.github.io/libgrape-lite)

## License

**libgrape-lite** is distributed under [Apache License 2.0](./LICENSE). Please note that third-party libraries may not have the same license as libgrape-lite.

## Acknowledgements

- [flat_hash_map](https://github.com/skarupke/flat_hash_map), an efficient hashmap implementation;
- [granula](https://github.com/atlarge-research/granula), a tool for gathering performance information for LDBC Benchmark;
- [cppkafka](https://github.com/mfontanini/cppkafka), a C++ wrapper of librdkafka;
- [xoroshiro](http://prng.di.unimi.it/), a pseudo-random number generator.


## Publications
- Wenfei Fan, Jingbo Xu, Wenyuan Yu, Jingren Zhou, Xiaojian Luo, Ping Lu, Qiang Yin, Yang Cao, and Ruiqi Xu. Parallelizing Sequential Graph Computations. ACM Transactions on Database Systems (TODS) 43(4): 18:1-18:39.

- Wenfei Fan, Jingbo Xu, Yinghui Wu, Wenyuan Yu, Jiaxin Jiang. GRAPE: Parallelizing Sequential Graph Computations. The 43rd International Conference on Very Large Data Bases (VLDB), demo, 2017 (**the Best Demo Award**).

- Wenfei Fan, Jingbo Xu, Yinghui Wu, Wenyuan Yu, Jiaxin Jiang, Zeyu Zheng, Bohan Zhang, Yang Cao, and Chao Tian. Parallelizing Sequential Graph Computations, ACM SIG Conference on Management of Data (SIGMOD), 2017 (**the Best Paper Award**).


## Getting involved

- Read [contribution guide](./CONTRIBUTING.md).
- Join in the [Slack channel](https://join.slack.com/t/graphscope/shared_invite/zt-fo88h3o7-4FGkoEFuzSBxmkGxOriPTw)
- Please report bugs by submitting a GitHub issue.
- Submit contributions using pull requests.

Thank you in advance for your contributions!
