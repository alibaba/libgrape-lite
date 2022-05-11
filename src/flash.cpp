#include "core/flash.h"

void update_info(const char *st, bool &directed, bool &bipartite, bool &weighted) {
	if(strcmp(st, "directed") == 0 ) directed = true;
	if(strcmp(st, "bipartite") == 0 ) bipartite = true;
	if(strcmp(st, "weighted") == 0 ) weighted = true;
}

void show_graph_w(string path) {
	int n, maxd, nx;
	bool weighted;
	GFS::get_graph_info(path, n, maxd, nx, weighted);
	int *adj = new int[maxd];
	float *adj_w = new float[maxd];

	string path_idx = path + "graph.idx";
	string path_dat = path + "graph.dat";
	string path_w = path + "graph.w";

	MyReadFile file_idx(path_idx, BUFFERED);
	MyReadFile file_dat(path_dat, BUFFERED);
	MyReadFile file_w(path_w, BUFFERED);

	for(int i = 0; i < n; ++i) {
		int din;
		int len = GFS::load_nbr(file_dat, file_idx, i, adj, din, false);
		GFS::load_w(file_w, file_idx, i, adj_w, false);
		printf( "%d:", i);
		for(int j = 0; j < len; ++j)
			printf( " (%d,%0.3f)", adj[j], adj_w[j] );
		printf( "\n" );
	}

	delete[] adj;
	delete[] adj_w;
}

int main(int argc, char *argv[]) {
	Graph<>::initialize();
	int n_proc;
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if( id == 0 ) {
		printf( "argc=%d\n", argc );
		for( int i = 0; i < argc; ++i )
			printf( "argv[%d]=%s\n", i, argv[i] );

		setvbuf(stdout, NULL, _IONBF, 0);
		setvbuf(stderr, NULL, _IONBF, 0);
		printf( "start\n" );
	}

	long t = clock();

	if( argc > 1 ) {
		if( strcmp( argv[1], "format" ) == 0 ) {
			//mpirun -n 4 -hostfile host_file ./biggraph format /projects2/NNSGroup/lqin/dataset/twitter-2010/ /scratch/ twitter-2010
			string path_bin = argv[2];
			string path_gfs = argv[3];
			string dataset = argv[4];
			Graph<>::format(path_gfs, dataset, path_bin);
		} else if( strcmp( argv[1], "txt2bin" ) == 0 ) {
			//mpirun ./biggraph txt2bin /projects2/NNSGroup/lqin/dataset/twitter-2010/ /scratch/ /projects2/NNSGroup/lqin/dataset/twitter-2010/
			if( id == 0 ) {
				//printf( "procesing txt2bin...\n" );
				string path_txt = argv[2];
				string path_tmp = argv[3];
				string path_bin = argv[4];
				bool directed = false, bipartite = false, weighted = false;
				if(argc > 5) update_info(argv[5], directed, bipartite, weighted);
				if(argc > 6) update_info(argv[6], directed, bipartite, weighted);
				if(argc > 7) update_info(argv[7], directed, bipartite, weighted);

				if(weighted) GFS::txt2bin_w(path_txt, path_tmp, path_bin, directed, bipartite);
				else GFS::txt2bin(path_txt, path_tmp, path_bin, directed, bipartite);

				int unused __attribute__((unused)) = system((string("rm ") + path_tmp + string("edges_*")).c_str());
			}
		}
	}

	if( argc <= 1) {
	}

	MPI_Barrier(MPI_COMM_WORLD);
	t = clock() - t;
	if( id == 0 ) {
		printf( "end\n" );
		printf( "total time=%0.3lf secs\n", t*1.0/CLOCKS_PER_SEC);
	}
	Graph<>::finalize();

	return 0;
}
