# GNN sampler

**libgrape-lite** follows a flexible *modular* and *header-only* design. In addition to analytical apps, other applications and components could also be easily developed and plugged in. As an example, we developed a simple graph sampler for online/offline GNN training/inference, by customizing a new AppendOnlyEdgeCutFragment that supports adding new edges/vertices as updates, and integrating Kafka into the main loop for online updates and queries.

Most of GNN models follow a neighborhood aggregation strategy, where each vertex iteratively updates its representation by aggregating representations of its neighbors. For a GNN model with `L` layers, each vertex needs to know its all neighbors within `L` hops as well as their feature information. Many real-world graphs have highly skewed power-law degree distributions, and some vertices may have very large degrees, causing the scalability problem. To solve this problem, since [GraphSage](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf), various sampling techniques have been introduced into GNN models, by down-sampling the neighbors of each vertex. In GNN training phase, each vertex only utilizes a fixed-size set of neighbors, instead of the full set of neighbors.


Built with **libgrape-lite**, this example implements a sampler that supports the following three built-in GNN neighbor sampling strategies:

- Random sampling: each vertex randomly chooses neighbors;
- EdgeWeight sampling: each vertex randomly chooses neighbors based on the edge weight distribution;
- Top-K sampling: each vertex chooses K neighbors with top-K edge weights.

## Building the Sampler

The sampler can be built with the whole repo in the root directory, or built with the specific target.
```bash
make gnn_sampler
```

## Running the Sampler on Static Graph

### Graph format

The graph format is the same as the repo. See [Graph format](https://github.com/alibaba/libgrape-lite/blob/reorg/README.md).

### Sampling

To run a sampler on a static graph in local or on a cluster, users may use commands like these:

```bash
# run graph sampling in local, with random sampling strategy. Each vertex samples neighbors within 2 hops, 10 neigbors in each hop.
mpirun -n 4 ./run_sampler --vfile ../dataset/p2p-31.v --efile ../dataset/p2p-31.e --sampling_strategy random --hop_and_num 4-5 --out_prefix ./output_sampling

# or run sampling with 4 workers on a cluster with the same parameters.
mpirun -n 4 -hostfile HOSTFILE ./run_sampler --vfile ../dataset/p2p-31.v --efile ../dataset/p2p-31.e --sampling_strategy random --hop_and_num 4-5 --out_prefix ./output_sampling
```

### Parameters
As shown in the example command, the sampler receives 5 parameters:

- `vfile`: vertex file of input graph.
- `efile`: edge file of input graph.
- `sampling_strategy`: select a strategy, currently we support three built-in strategies: 'random', 'edge_weight' and 'top_k'.
- `hop_and_num`: the hop and the numbers of neighbors to sample. The value of this parameter is `n` numbers separated by '-', representing the number of sampled neighbors for the `n` hops. e.g., '4-5' means that each vertex samples neighbors within 2 hops. For the first hop, each vertex samples 4 neighbors, and for the second hop, each vertex samples 5 neighbors.
- `out_prefix`: ouput file prefix.

### Result

The sampling would work over all vertices in the graph. The format of output for each vertex looks like this:

```
sampling_node, 1st_hop_neighbors[v1, v2, ... vn], 2nd_hop_neighbors[u1, u2, ... un], ...

# in the above example, each line in the result would be looks like this
v, v_1_nb1, v_1_nb2, v_1_nb3, v_1_nb4, v_2_nb1, v_2_nb2, v_2_nb3, v_2_nb4, v_2_nb5
```

The result can be considered as a level-wise traversal of the sampling path tree.

## Sampling on Dynamic Graph (Append-Only)

**gnn_sampler** supports sampling on dynamic(append-only) graphs. We use
Kafka as the MQ to produce graph updates/queries and to ingest the sampling results.
Users can send the updates on graphs (in a format of edge triplet) and queries via Kafka to append the graph and to sample on vertices.

### Deploying Kafka

Users may obtain a [Kafka](https://archive.apache.org/dist/kafka/2.3.0/kafka_2.11-2.3.0.tgz) binary release and deploy it following these commands:

```bash
# first, start zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# then start the kafka server
bin/kafka-server-start.sh config/server.properties

# create a topic named 'sampling_input' for input
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic sampling_input

# create a topic named 'sampling_output' for output
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic sampling_output
```
Please refer to [Quick Start](https://kafka.apache.org/quickstart) provided by Kafka for more details.

### Message format

**gnn_sampler** recognizes two kinds of messages from the kafka input topic.

1. Message for graph update.
This kind of messages is in a format of edge triplet, i.e., (src, dst and the data on edge), prefixed with a char 'e'. For example: `e 0 1 3.75`.

2. Message for sampling vertex(as query).
This kind of messages contains a node that users want to sampling from, prefixed with a char 'q', for example: `q 0`.

### Running the sampler

Now users can run the sampler, with enabling the Kafka to generate updates/queries and to sink the sampling results. In addition to the launch command for static graphs, sampling on dynamic graphs needs several more flags to assign the broker, input_topic and output_topic. For example:

```bash
# run sampling on dynamic graph
mpirun -n 4 ./run_sampler --vfile ../dataset/p2p-31.v --efile ../dataset/p2p-31.e --sampling_strategy random --hop_and_num 10-10 --enable_kafka true --broker_list localhost:9092 --input_topic sampling_input --group_id consumer_xx --partition_num 1 --batch_size 100 --time_interval 10
--output_topic sampling_output
```

extra parameters:

- `broker_list`: list of kafka brokers, in format of 'server1:port,server2:port,...'.
- `enable_kafka`: enable kafka.
- `input_topic`: the input topic for graph updates/queries.
- `group_id`: consumer group id.
- `partition_num`: partition num of input topic.
- `batch_size`: the batch size of queries from the input topic.
- `time_interval`: the timeout interval waiting to be batched (by second).
- `output_topic`: the output topic for sampling results.


### Producing or consuming messages with script

You may want to use scripts provided by Kafka to produce/consume messages for testing.

```bash
# produce example
bin/kafka-console-producer.sh --broker-list localhost:9092 --topic sampling_input
> e 0 1 1
> e 0 2 2
> q 0
> q 1

# consume sampling_output topic from beginning
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic sampling_output --from-beginning
```
