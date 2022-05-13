# Step 1 Prepare Machines
Suppose we use 4 Machines named Orion1, Orion2, Orion3, and Orion4, and Orion1 is the master. Each machine has a local disk with location: `/scratch/gfs/` and Orion1 has a local disk with location: `/scratch/tmp/`
    
# Step 2 Compile Flash
Suppose we put the Flash Folder into a shared folder with location /myname/
Go to folder `/myname/flash/run/` and execute the command: 

`./compile-flash.sh`

# Step 3 Prepare Dataset on the Master Machine (Orion1)
Suppose we use the orkut dataset under the folder `/myname/dataset/orkut/`
## Step 3.1
Download the txt file from [snap](http://snap.stanford.edu/data/com-Orkut.html) and extract it to `/myname/dataset/orkut/` on Orion1, and rename the txt file to graph.txt. Now we have  `/myname/dataset/orkut/graph.txt`
## Step 3.2 
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
    
# Step 4: Upload the Graph to the Distributed File System
Suppose we want to run 8 processes, i.e., each machine has 2 processes. 
## Step 4.1 
Edit `/myname/flash/run/host_file` to be the `machine_name:1` on each line of `host_file` (see the sample)
## Step 4.2
Goto folder `/myname/flash/run/` and execute the command: 

`./format.sh 8 /myname/dataset/orkut/ /scratch/gfs/ orkut`
    
# Step 5: Execute Flash Algorithm 
## Step 5.1
Write a flash c++ algorithm under folder `/myname/flash/src/apps/`, for example `/myname/flash/src/apps/bfs.cpp`
## Step 5.2
Goto folder /myname/flash/run/ and compile the algorithm using 

`./compile.sh bfs`
## Step 5.3 
Goto folder /myname/flash/run/ and run the algorithm using: 

`mpirun -n 8 -hostfile host_file ./bfs /scratch/gfs/ orkut 1`

Here, the last 1 is a parameter used in bfs.cpp, which is the start node id for BFS.
    
