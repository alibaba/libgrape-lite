<h1 align="center">
    <br> FLASH
</h1>
<p align="center">
   A Framework for Programming Distributed Graph Processing Algorithms
</p>

**FLASH** is a distributed framework for programming a broad spectrum of graph algorithms, including clustering, centrality, traversal, matching, mining, etc. Based on its high-level interface and an efficient system implementation, FLASH achieves good expressiveness, productivity and efficiency at the same time. FLASH follows the vertex-centric philosophy, but it moves a step further for stronger expressiveness by providing flexible control flow, the operations on arbitrary vertex sets and beyond-neighborhood communication. FLASH makes diverse complex graph algorithms easy to write at the distributed runtime. The algorithms expressed in FLASH take only a few lines of code, and provide a satisfactory performance.

## Dependencies
**FLASH** is developed and tested on Red Hat 4.8.5. It should also work on other unix-like distributions. Building FLASH requires the following softwares installed as dependencies.
- A C++ compiler supporting OpenMP and C++11 features
- MPICH (>= 2.1.4) or OpenMPI (>= 3.0.0)


## Building Steps

### Step 1 Prepare Machines
Suppose we use 4 Machines named Orion1, Orion2, Orion3, and Orion4, and Orion1 is the master. Each machine has a local disk with location: `/scratch/gfs/` and Orion1 has a local disk with location: `/scratch/tmp/`
    
### Step 2 Compile FLASH
Suppose we put the FLASH Folder into a shared folder with location /myname/
Go to folder `/myname/flash/run/` and execute the command: 

`./compile-flash.sh`

## Preparing Graph Data

### Step 1 Prepare Dataset on the Master Machine (Orion1)
Suppose we use the orkut dataset under the folder `/myname/dataset/orkut/`
#### Step 1.1
Download the txt file from [snap](http://snap.stanford.edu/data/com-Orkut.html) and extract it to `/myname/dataset/orkut/` on Orion1, and rename the txt file to graph.txt. Now we have  `/myname/dataset/orkut/graph.txt`
#### Step 1.2 
Goto folder /myname/flash/run/ and execute the following command for an undirected graph: 

`./txt2bin.sh /myname/dataset/orkut/ /scratch/tmp/`

or execute the following command for a directed graph

`./txt2bin.sh /myname/dataset/orkut/ /scratch/tmp/ directed`

or execute the following command for a weighted graph 

`./txt2bin.sh /myname/dataset/orkut/ /scratch/tmp/ weighted`

or execute the following command for a bipartite graph 

`./txt2bin.sh /myname/dataset/orkut/ /scratch/tmp/ bipartite`

Note that the keywords directed, weighted, and bipartite can be combined arbitrarily according to the graph type.
Now the text file will be stored as binary files on the Master Machine
    
### Step 2: Upload the Graph to the Distributed File System
Suppose we want to run 8 processes, i.e., each machine has 2 processes. 
#### Step 1.1 
Edit `/myname/flash/run/host_file` to be the `machine_name:1` on each line of `host_file` (see the sample)
#### Step 2.2
Goto folder `/myname/flash/run/` and execute the command: 

`./format.sh 8 /myname/dataset/orkut/ /scratch/gfs/ orkut`

## Running FLASH Applications
We have provided some example applications in `/myname/flash/src/apps/` ， and more applications are continuously being added. The following steps are required for adding and executing a new algorithm in FLASH. 

### Step 1: Write a algorithm
Write a FLASH c++ algorithm under folder `/myname/flash/src/apps/`, for example `/myname/flash/src/apps/bfs.cpp`
### Step 2: Compile the algorithm
Goto folder /myname/flash/run/ and compile the algorithm using 

`./compile.sh bfs`
### Step 3: Execute 
Goto folder /myname/flash/run/ and run the algorithm using: 

`mpirun -n 8 -hostfile host_file ./bfs /scratch/gfs/ orkut 1`

Here, the last 1 is a parameter used in bfs.cpp, which is the start node id for BFS.
    
