SHELL = /bin/bash
BIN_PATH = /tmp/tmp.oOqUdB4akj/cmake-build-debug-remote-host/run_app
DATASET_NAME = ../thirdparty/libgrape-lite/dataset/p2p-31
TMP_DIR = /tmp/grape_gpu
SSSP_SOURCE = 6
BFS_SOURCE = 6

RUN_APP = run_app() {                                                                        \
  app=$$1                                                                                    \
  directed=$$2                                                                               \
  lb=$$3;                                                                                    \
  ngpu=$$4;                                                                                  \
  out_filename=$$5;                                                                          \
  if [[ $$directed == "directed" ]]; then                                                    \
	is_directed="true";                                                                      \
  elif [[ $$directed == "undirected" ]]; then                                                \
	is_directed="false";                                                                     \
  else                                                                                       \
	exit 1;                                                                                  \
  fi;                                                                                        \
  mkdir -p $(TMP_DIR);                                                                       \
  mpirun -n $$ngpu $(BIN_PATH)                                                               \
     -application=$$app                                                                      \
     -directed=$$is_directed                                                                 \
     -efile=$(DATASET_NAME).e                                                                \
     -vfile=$(DATASET_NAME).v                                                                \
     -lb=$$lb                                                                                \
     -out_prefix=$(TMP_DIR)                                                                  \
     -bfs_source=$(SSSP_SOURCE)                                                              \
     -sssp_source=$(BFS_SOURCE)                                                              \
     -pr_mr=30 -pr_d=0.85                                                                    \
     -cdlp_mr=10;                                                                            \
  cat $(TMP_DIR)/* | sort -n > $$out_filename && rm -rf $(TMP_DIR);                          \
}

TARGETS = gen_targets() { \
    for lb in cta cm cmold wm strict none; do \
      echo "p2p-31-$${1}-$${2}-$${lb}-1"; \
      echo "p2p-31-$${1}-$${2}-$${lb}-2"; \
    done; \
}

all: pr.verified sssp.verified bfs.verified wcc.verified lcc.verified cdlp.verified
	@:

# stem: p2p-31-directed-app-lb-ngpu, e.g. p2p-31-directed-sssp-cm-1
p2p-31-%:
	out_filename=$@;                                                                         \
	directed=$$(echo $* | cut -d"-" -f1,1);                                                  \
	app=$$(echo $* | cut -d"-" -f2,2);                                                       \
	lb=$$(echo $* | cut -d"-" -f3,3);                                                        \
	ngpu=$$(echo $* | cut -d"-" -f4,4);                                                      \
	$(RUN_APP);                                                                              \
	run_app $$app $$directed $$lb $$ngpu $$out_filename;                                     \
	if [[ $$app == "bfs" || $$app == "dobfs" ]]; then                                        \
	  ./sssp_bfs.py $(DATASET_NAME).e $$out_filename $$directed $(BFS_SOURCE) bfs;           \
  	elif [[ $$app == "sssp" ]]; then                                                         \
  	  ./sssp_bfs.py $(DATASET_NAME).e $$out_filename $$directed $(SSSP_SOURCE) sssp;         \
  	elif [[ $$app == "pagerank" || $$app == "pagerank_pull" ]]; then                         \
  	  ./pr.py $(DATASET_NAME).e $$out_filename $$directed;                                   \
  	elif [[ $$app == "wcc" ]]; then                                                          \
  	  ./wcc.py $(DATASET_NAME).e $$out_filename;                                             \
  	elif [[ $$app == "lcc" ]]; then                                                          \
  	  ./lcc.py $(DATASET_NAME).e $$out_filename;                                             \
  	elif [[ $$app == "cdlp" ]]; then                                                         \
  	   ./cdlp.py $(DATASET_NAME).e $$out_filename;                                            \
  	else                                                                                     \
  	  exit 1;                                                                                \
  	fi;                                                                                      \
	if [[ $$? -ne 0 ]]; then 							                                     \
	  exit 1; 											                                     \
	fi;

bfs.verified: $(shell $(TARGETS); gen_targets directed bfs; gen_targets undirected bfs)

dobfs.verified: $(shell $(TARGETS); gen_targets directed dobfs; gen_targets undirected dobfs)

sssp.verified: $(shell $(TARGETS); gen_targets directed sssp; gen_targets undirected sssp)

pr.verified: $(shell $(TARGETS); gen_targets directed pagerank; gen_targets undirected pagerank)

pr_pull.verified: $(shell $(TARGETS); gen_targets directed pagerank_pull; gen_targets undirected pagerank_pull)

wcc.verified: $(shell $(TARGETS); gen_targets undirected wcc)

lcc.verified: $(shell $(TARGETS); gen_targets undirected lcc)

cdlp.verified: $(shell $(TARGETS); gen_targets undirected cdlp)

.PHONY: clean all

clean:
	rm -f p2p-31* *verified