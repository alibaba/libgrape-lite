## GUM-artifact
GUM is a CUDA library for parallel graph processing on GPU. It focuses on the dynamic load balance (DLB) problem and the long tail *(LT) problem.

## Dependencies
- [CMake](https://cmake.org/download/) (>2.8)
- [MPICH](https://www.mpich.org/) (>=2.1.4) or [OpenMPI](https://www.open-mpi.org/) (>=3.0.0)
- [Boost](https://www.boost.org/) (>1.58)
- [glogs](https://github.com/google/glog) (>=0.3.4)
- [gflags](https://github.com/gflags/gflags) (>=2.2.0)
- [OR-tools](https://github.com/google/or-tools) (>=9.1)
- [cub](https://github.com/NVIDIA/cub) (>=1.15)
- [moderngpu](https://github.com/moderngpu/moderngpu) (>=2.12)
- [thrust](https://github.com/NVIDIA/thrust) (>=1.16)
- [SNAP datasets] (https://snap.stanford.edu/data/index.html)

## Installation

Once the required dependencies have been installed, go to the root directory and do a out-of-source build using CMake.

```bash
mkdir build && cd build
cmake ..
make -j
```

## Example

**GUM** support BFS, SSSP, PR, WCC, LCC, CDLP algorithms and it accepts market format graph. Suppose your data is under `~/dataset`, to run bfs application with:

`mpirun --allow-run-as-root -n 8 ./build/run_app -application bfs --partitioner seg --lb=none -bfs_source=6 -efile ~/dataset/road_usa/road_usa.mtx`

- `application` : Optional parameter to indicate the algorithm.
- `efile` :Path to the input edge file.
- `partitioner` : Optional parameter to set the partitioner.
- `lb`: Optional parameter to set the warp scheduling policy.
- `serialization_prefix` : Optional parameter to cache the input graph.

## License
GUM is distributed under Apache License 2.0. Please note that third-party libraries may not have the same license as GUM.
