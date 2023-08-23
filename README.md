<h1 align="center">
    <br> Ingress
</h1>
<p align="center">
   Automating Incremental Graph Processing with Flexible Memoization
</p>

## 1. What is it?

Ingress is an automated system for incremental graph processing.It is able to incrementalize batch vertex-centric algorithms into their incremental counterparts as a whole, without the need of redesigned logic or data structures from users. Underlying Ingress is an automated incrementalization framework equipped with four different memoization policies, to support all kinds of vertex-centric computations with optimized memory utilization. [[Read more]](https://doi.org/10.14778/3461535.3461550.pdf)


##  2. Building **Ingress**

### 2.1 Dependencies

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
- [Apache Kafka](https://github.com/apache/kafka) (>= 2.3.0);
- [librdkafka](https://github.com/edenhill/librdkafka)(>= 0.11.3);
- [ANTLR4](https://github.com/antlr/antlr4)(=4.9.2)
- [Z3](https://github.com/Z3Prover/z3)(>= 4.8.18)

### 2.2 Building Ingress and examples
You can also refer to hello_auto_ingress.md to configure ANTLR4 and Z3 dependencies.Once the required dependencies have been installed, go to the root directory of Ingress and do a out-of-source build using CMake.

```bash
mkdir build && cd build
cmake ..
make ingress
```

### 2.3 Running Ingress applications

**Ingress** provides seven algorithms from the LDBC benchmark as examples. The deterministic algorithms are, single-source shortest path(SSSP), connected component(CC), PageRank,and breadth first search(BFS),PHP,SSWP,GCN.

```bash
# Run the pagerank algorithm with the automatic selection engine
$   cd ./grape/examples/analytical_apps/antlr/src
$   ./eng.sh

# Run the sssp algorithm with the automatic selection engine
$   cd /Ingress-for_expr_suminc_by_ys/examples/analytical_apps/antlr/src
$   vim ./eng.sh
# Change your algorithm location like ../../../sssp/sssp_ingress.h and change your command-line parameters like sssp
$ ./eng.sh
```
 The executable takes the following command-line parameters:
 - `-application` : Optional parameter to indicate the algorithm. 
 - `-vfile` : Path to the input vertex file.
 - `-efile` :Path to the input edge file.
 - `-nEdges` : Number of edge operations to be processed in a given update batch.
 - `-outputFile` : Optional parameter to print the output of a given algorithms.
 - Input graph file path (More information on the input format can be found in [Section 2.4](#24-graph-input-and-stream-input-format)).

### 2.4 Graph Input and Stream Input Format

The initial input graph should be in the [adjacency graph format](http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html). 
For example, the efile format (edgelist) and the vfile format for a sample graph are shown below.

efile format:
```txt
0 1
0 2
2 0
2 1
```
vfile format:
```txt
0
1
2
```
