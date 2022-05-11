#ifndef GRAPH_H_
#define GRAPH_H_

#include <mpi.h>

#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <climits>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <typeinfo>

#include "gfs.h"
#include "type.h"
#include "buffer.h"
#include "atts.h"
#include "reduce.h"

using namespace std;

#define NBR(I) v_all[info.adj[I]&ALL]

#define dst_id info.id
#define v_id info.id
#define v_deg info.deg
#define v_din info.din
#define v_dout info.dout

#define get_nb_id(I) (info.get_nbr_id(I))
#define get_in_id(I) (info.get_nbr_id(I))
#define get_out_id(I) (info.get_nbr_id(info.din+I))
#define get_v(I) v_all[I]
#define GetV(I) G.v_all[I]
#define get(I) v_all[I]
#define put(v, d) v.cp_from(d)

#define for_nb(...) {info.load_nbr(); for(int I=0;I<info.deg;I++) {VTYPE &nb=NBR(I); __VA_ARGS__;}}
#define for_in(...) {info.load_nbr(); for(int I=0;I<info.din;I++) {VTYPE &nb=NBR(I); __VA_ARGS__;}}
#define for_out(...) {info.load_nbr(); for(int I=info.din;I<info.deg;I++) {VTYPE &nb=NBR(I); __VA_ARGS__;}}
#define for_i(...) for(int i=0;i<len;++i){ __VA_ARGS__;}
#define for_j(...) for(int j=0;j<len;++j){ __VA_ARGS__;}
#define for_k(...) for(int k=0;k<len;++k){ __VA_ARGS__;}

#define nb_id (info.adj[I]&ALL)
#define nb_w (info.get_nbr_w(I))

#define n_vertex G.n
#define n_x G.nx
#define SetDataset2(path,dataset) GRAPH G(path, dataset); VSet::g = &G; G.all_nodes(All.s); G.t=clock();
#define SetDataset3(path,dataset,in_mem) GRAPH::edge_in_mem=in_mem; GRAPH G(path, dataset); VSet::g = &G; G.all_nodes(All.s); G.t=clock();

#define GetSetDataset(_1, _2, _3, NAME, ...) NAME
#define SetDataset(...) GetSetDataset(__VA_ARGS__, SetDataset3, SetDataset2, _1, ...) (__VA_ARGS__)
#define SetArbitraryPull(B) GRAPH::arbitrary_pull=(B)
#define SetSynAll(B) SetArbitraryPull(B)

#define in_master G.master()
#define GetTime G.get_time
#define print(...) {if(in_master) printf(__VA_ARGS__); else sprintf(G.print_buf.data(),__VA_ARGS__);}

TUPLE1(Integer,int,val);

template<class VTYPE=Integer> class Graph {
public:
	string dataset;				//e.g. twitter-2010
	string path;				//e.g. /scratch/data/

public:
	static int id;				//the id of the processor
	static int n_procs;			//number of processors
	static bool is_master;		//whether id == 0

public:
	static int critical_atts;
	static long t;
	static bool edge_in_mem;
	static bool arbitrary_pull;
	static vector<char> print_buf;

private:
	MetaInfo info;
	int *con_dat;
	int **con;

	float *con_w_dat;
	float **con_w;

private:
	void send_to_neighbor(int atts);
	void update_buf(BufManager &b, int atts);

	bool next_all_bm(int &vid, int &cid);

	//in push, if the receiver receive the message from mirror vertices, then start_id should be the cid of the receiver, not the sender
	void init_all_bm(int start_id=-1);
	void set_info(int u);
	void send_buf(const vector<int> &list_change, int atts_local, bool syn_all=false);

public:
	void pull(function<void(VTYPE&, VTYPE*&, MetaInfo&)> f_pull, const vector<int> &list_v, vector<int> &list_result, int atts=-1);
	void push(function<void(VTYPE&, VTYPE&, VTYPE*&, MetaInfo&)> f_push, const vector<int> &list_v, vector<int> &list_result);

	void push(function<void(VTYPE&, VTYPE&, VTYPE*&, VTYPE*&, MetaInfo&)> f_cmb, function<void(VTYPE&, VTYPE&, VTYPE*&, MetaInfo&)> f_agg,
			const vector<int> &list_v, vector<int> &list_result, int atts_agg=-1, int atts_cmb=-1);

	void local(function<void(VTYPE&, VTYPE*&, MetaInfo&)> f_local, const vector<int> &list_v, int atts=-1);
	void gather(function<void(VTYPE&, VTYPE*&, MetaInfo&)> func, const vector<int> &list_v, int atts=-1);
	void traverse(function<void(VTYPE&, VTYPE*&, MetaInfo&)> func);
	void filter(function<bool(VTYPE&, VTYPE*&, MetaInfo&)> f_filter, const vector<int> &list_v, vector<int> &list_result);
	void all_nodes(vector<int> &list_v);

private:
	static int append_idx(FILE *wf_dat, FILE *wf_idx, FILE *wf_inf, fileint &wf_pos, char *buf, int &max_deg, char *nb_ids);
	static int append_idx_w(FILE *wf_w, char *buf_w);

public:
	//e.g., format("/scratch/graph/", "twitter-2010", "/projects2/NNSGroup/lqin/dataset/twitter-2010/");
	//the bin folder should be in the master
	static void format(string path, string dataset, string path_bin);
	static void initialize() {
		int flag = 0;
		MPI_Initialized(&flag);
		if(!flag) MPI_Init(NULL, NULL);
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
		is_master = (id == MASTER);
	}

	static bool master() {return is_master;}
	static void finalize() {MPI_Finalize();}
	static double get_time() {return (clock()-t) *1.0/CLOCKS_PER_SEC;}

public:
	//e.g., Graph("/scratch/", "twitter-2010");
	Graph(string path, string dataset);
	~Graph();

public:
	int n, n_local, max_deg_local, nx;
	VTYPE *v_all;
	bool weighted;

private:
	VTYPE *v_loc, *v_cmb, *v_loc_tmp;
	int *adj;
	float *adj_w;
	MyReadFile f_idx, f_dat, f_w;
	BufManager *bm, b_tmp;

	char *nb_ids_dat;
	char **nb_ids;

	unsigned *bset;
	int *deg, *din;

private:
	VTYPE v_tmp;

public:
	inline int get_n() {return n;}
	inline int get_n_local() {return n_local;}
};

//implementation
template <class VTYPE> int Graph<VTYPE>::id = 0;
template <class VTYPE> bool Graph<VTYPE>::edge_in_mem = false;
template <class VTYPE> long Graph<VTYPE>::t = 0;
template <class VTYPE> int Graph<VTYPE>::critical_atts = -1;
template <class VTYPE> int Graph<VTYPE>::n_procs = 0;
template <class VTYPE> bool Graph<VTYPE>::is_master = false;
template <class VTYPE> bool Graph<VTYPE>::arbitrary_pull = false;
template <class VTYPE> vector<char> Graph<VTYPE>::print_buf = vector<char>(4096);

template<class VTYPE> Graph<VTYPE>::Graph(string path, string dataset) {
	int flag = 0;
	MPI_Initialized(&flag);
	if( !flag ) MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
	is_master = (id == MASTER);

	if(path.c_str()[path.size()-1] != '/') path=path+"/";

	this->dataset = dataset;
	this->path = path;

	char tmp[64];
	sprintf(tmp, "_%d_%d", n_procs, id);
	f_idx.fopen(path + dataset + string(tmp) + ".idx", BUFFERED);
	f_dat.fopen(path + dataset + string(tmp) + ".dat", BUFFERED);
	weighted = f_w.fopen(path + dataset + string(tmp) + ".w", BUFFERED);

	f_idx.fread(&n, sizeof(int));

	v_all = new VTYPE[n];
	n_local = NLOC(id); //n/n_procs + (id<(n%n_procs)?1:0);
	v_loc = new VTYPE[n_local];
	v_loc_tmp = new VTYPE[n_local];
	nb_ids_dat = new char[((n_procs+7)/8) * n_local];
	nb_ids = new char*[n_local];

	deg = new int[n_local];
	din = new int[n_local];

	MyReadFile f_inf(path + dataset + string(tmp) + ".inf", BUFFERED);

	int p = 0;
	for( int i = 0; i < n_local; ++i ) {
		nb_ids[i] = nb_ids_dat + p;
		p += (n_procs+7)/8;
		f_inf.fread(nb_ids[i], (n_procs+7)/8);
	}
	f_inf.fread(&max_deg_local, sizeof(int));
	f_inf.fread(&nx, sizeof(int));
	f_inf.fclose();

	long long pre_pos, now_pos, m_local = 0;
	f_idx.fread(&pre_pos, sizeof(long long));

	for(int i = 0; i < n_local; ++i) {
		f_idx.fread(&din[i], sizeof(int));
		f_idx.fread(&now_pos, sizeof(long long));
		deg[i] = (int)((now_pos-pre_pos)/sizeof(int));
		m_local += deg[i];
		pre_pos = now_pos;
	}

	if(edge_in_mem) {
		con_dat = new int[m_local];
		con = new int*[n_local];
		now_pos = 0;
		for(int i = 0; i < n_local; ++i) {
			f_dat.fread(con_dat+now_pos, sizeof(int) * deg[i]);
			con[i] = con_dat+now_pos;
			now_pos += deg[i];
		}
		if(weighted) {
			con_w_dat = new float[m_local];
			con_w = new float*[n_local];
			now_pos = 0;
			for(int i = 0; i < n_local; ++i) {
				f_w.fread(con_w_dat+now_pos, sizeof(float) * deg[i]);
				con_w[i] = con_w_dat+now_pos;
				now_pos += deg[i];
			}
		}
	} else {
		con_dat = NULL; con = NULL;
		con_w_dat = NULL; con_w = NULL;
	}

	adj = new int[max_deg_local];
	adj_w = NULL;
	if(weighted) adj_w = new float[max_deg_local];

	bm = new BufManager[n_procs];
	for( int i = 0; i < n_procs; ++i )
		bm[i].set_info(n, n_procs, i);

	bset = new unsigned[(n+31)/32];
	memset(bset, 0, sizeof(unsigned) * (n+31)/32);

	v_cmb = NULL;
	info.n = n;
	info.n_procs = n_procs;
	info.cid = id;
	info.f_dat = &f_dat;
	info.f_idx = &f_idx;
	info.f_w = &f_w;
	info.adj = adj;
	info.adj_w = adj_w;
	info.bset = bset;
	info.deg_all = deg;
	info.din_all = din;
	info.con = con;
	info.con_w = con_w;
	info.edge_in_mem = edge_in_mem;

	if(id == 0 ) printf( "dataset=%s, n=%d, n_procs=%d, edge_in_mem=%s\n", (path+dataset).c_str(), n, n_procs, edge_in_mem?"true":"false" );
}

template<class VTYPE> Graph<VTYPE>::~Graph() {
	f_idx.fclose();
	f_dat.fclose();
	if(weighted) f_w.fclose();
	delete[] v_all;
	delete[] adj;
	delete[] bm;
	delete[] nb_ids;
	delete[] nb_ids_dat;
	delete[] v_loc;
	delete[] v_loc_tmp;
	delete[] bset;
	delete[] deg;
	delete[] din;

	if(adj_w) delete[] adj_w;
	if(!con) delete[] con;
	if(!con_dat) delete[] con_dat;
	if(!con_w) delete[] con_w;
	if(!con_w_dat) delete[] con_w_dat;
	if(!v_cmb) delete[] v_cmb;
	MPI_Finalize();
}

template<class VTYPE> void Graph<VTYPE>::format(string path, string dataset, string path_bin) {
	int n, max_deg, nx;
	if(path.c_str()[path.size()-1] != '/') path=path+"/";
	if(path_bin.c_str()[path_bin.size()-1] != '/') path_bin=path_bin+"/";
	bool weighted;

	if(is_master) {
		GFS::get_graph_info(path_bin, n, max_deg, nx, weighted);
		printf( "n=%d,max_deg=%d,nx=%d,weighted=%d\n", n, max_deg, nx, weighted?1:0);
	}

	MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&max_deg, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&nx, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&weighted, 1, MPI_CHAR, MASTER, MPI_COMM_WORLD);

	fileint now_pos = 0;
	char tmp[64];
	sprintf(tmp, "_%d_%d", n_procs, id);

	GFS::to_path(path); GFS::to_path(path_bin);
	string file_gfs = path + dataset + string(tmp);

	int buf_len = MAXBUF+(max_deg+1)*sizeof(int);

	FILE *wf_idx = fopen((file_gfs + ".idx").c_str(), "wb");
	fwrite(&n, sizeof(int), 1, wf_idx);
	fwrite(&now_pos, sizeof(fileint), 1, wf_idx);

	FILE *wf_dat = fopen((file_gfs + ".dat").c_str(), "wb");
	FILE *wf_inf = fopen((file_gfs + ".inf").c_str(), "wb");
	FILE *wf_w = NULL;
	if(weighted) wf_w = fopen((file_gfs + ".w").c_str(), "wb");

	int max_deg_local = 0;
	char *nb_ids = new char[(n_procs+7)/8];

	if(is_master) {
		string path_idx = path_bin + "graph.idx";
		string path_dat = path_bin + "graph.dat";
		string path_w = path_bin + "graph.w";

		MyReadFile file_idx(path_idx, BUFFERED);
		MyReadFile file_dat(path_dat, BUFFERED);
		MyReadFile file_w;
		if(weighted) file_w.fopen(path_w, BUFFERED);

		char** bufs = new char*[n_procs];
		int* pos = new int[n_procs];
		memset( pos, 0, sizeof(int) * n_procs);
		for(int i = 0; i < n_procs; ++i)
			bufs[i] = new char[buf_len];

		char **bufs_w = NULL;
		if(weighted) {
			bufs_w = new char*[n_procs];
			for(int i = 0; i < n_procs; ++i)
				bufs_w[i] = new char[buf_len];
		}

		printf( "[%d] Sending ...\n", id );
		for(int u = 0; u < n; ++u) {
			int p = u % n_procs, din;
			int len = GFS::load_nbr(file_dat, file_idx, u, bufs[p] + pos[p] + sizeof(int), din, false);
			if(weighted) GFS::load_w(file_w, file_idx, u, bufs_w[p] + pos[p] + sizeof(int), false);

			memcpy(bufs[p] + pos[p], &len, sizeof(int));
			if(weighted) memcpy(bufs_w[p] + pos[p], &len, sizeof(int));

			pos[p] += sizeof(int) + len * sizeof(int);
			if( u % 1000000 == 0 ) printf( "[%d] %d/%d\n", id, u, n );

			if(pos[p] >= MAXBUF || u >= n-n_procs) {
				int flag = u >= n-n_procs ? BUFEND : BUFCONT;
				memcpy(bufs[p] + pos[p], &flag, sizeof(int));
				if(weighted)  memcpy(bufs_w[p] + pos[p], &flag, sizeof(int));

				pos[p] += sizeof(int);
				printf( "[%d] Send %d data to node %d\n", id, pos[p], p);

				if( p != MASTER ) {
					MPI_Send(bufs[p], pos[p], MPI_CHAR, p, 0, MPI_COMM_WORLD);
					if(weighted) MPI_Send(bufs_w[p], pos[p], MPI_CHAR, p, 0, MPI_COMM_WORLD);
				} else {
					append_idx(wf_dat, wf_idx, wf_inf, now_pos, bufs[p], max_deg_local, nb_ids);
					if(weighted) append_idx_w(wf_w, bufs_w[p]);
				}

				pos[p] = 0;
			}
		}

		file_idx.fclose();
		file_dat.fclose();
		if(weighted) file_w.fclose();

		for(int i = 0; i < n_procs; ++i) delete[] bufs[i];
		delete[] bufs;

		if(weighted) {
			for(int i = 0; i < n_procs; ++i) delete[] bufs_w[i];
			delete[] bufs_w;
		}

		delete[] pos;
		printf( "[%d] Master Finish!\n", id );
	} else {
		printf( "[%d] Receiving ...\n", id );
		char *buf = new char[buf_len];
		char *buf_w = NULL;
		if(weighted) buf_w = new char[buf_len];

		MPI_Status status;
		int flag = BUFCONT;
		while(flag != BUFEND) {
			MPI_Recv(buf, buf_len, MPI_CHAR, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			flag = append_idx(wf_dat, wf_idx, wf_inf, now_pos, buf, max_deg_local, nb_ids);

			if(weighted) {
				MPI_Recv(buf_w, buf_len, MPI_CHAR, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				append_idx_w(wf_w, buf_w);
			}
		}
		delete[] buf;
		if(weighted) delete[] buf_w;
		printf( "[%d] Receiving Finished!\n", id );
	}

	fwrite(&max_deg_local, sizeof(int), 1, wf_inf);
	fwrite(&nx, sizeof(int), 1, wf_inf);

	fclose(wf_idx);
	fclose(wf_dat);
	fclose(wf_inf);
	if(weighted) fclose(wf_w);
	delete[] nb_ids;
}

template<class VTYPE> int Graph<VTYPE>::append_idx_w(FILE *wf_w, char *buf_w) {
	int len, pos = 0;
	memcpy(&len, buf_w + pos, sizeof(int)); pos += sizeof(int);

	while(len >= 0) {
		if(len) fwrite(buf_w+pos, sizeof(float), len, wf_w);
		pos += sizeof(float) * len;
		memcpy(&len, buf_w + pos, sizeof(int)); pos += sizeof(int);
	}
	return len;
}

template<class VTYPE> int Graph<VTYPE>::append_idx(FILE *wf_dat, FILE *wf_idx, FILE *wf_inf, fileint &wf_pos, char *buf, int &max_deg, char *nb_ids) {
	int len, pos = 0;
	memcpy(&len, buf + pos, sizeof(int)); pos += sizeof(int);

	while(len >= 0) {
		max_deg = max(max_deg, len);
		if(len) fwrite(buf+pos, sizeof(int), len, wf_dat);

		int din = 0;
		memset(nb_ids, 0, (n_procs+7)/8);
		for( int i = 0; i < len; ++i ) {
			int p;
			memcpy(&p, buf+pos+i*sizeof(int), sizeof(int));
			if(p&NEG) ++din;
			p &= ALL;
			p %= n_procs;
			nb_ids[p/8] |= 1<<(p%8);
		}
		if(wf_inf) fwrite(nb_ids, 1, (n_procs+7)/8, wf_inf);

		pos += sizeof(int) * len;
		wf_pos += sizeof(int) * len;
		fwrite(&din, sizeof(int), 1, wf_idx);
		fwrite(&wf_pos, sizeof(fileint), 1, wf_idx);
		memcpy(&len, buf + pos, sizeof(int)); pos += sizeof(int);
	}
	return len;
}

template<class VTYPE> void Graph<VTYPE>::filter(function<bool(VTYPE&, VTYPE*&, MetaInfo&)> f_filter, const vector<int> &list_v, vector<int> &list_result) {
	list_result.clear();
	for( size_t i = 0; i < list_v.size(); ++i ) {
		int u = list_v[i];
		set_info(u);
		if(f_filter(v_all[u], v_all, info)) list_result.push_back(u);
	}
}

template<class VTYPE> void Graph<VTYPE>::all_nodes(vector<int> &list_v) {
	list_v.clear();
	for(int i = 0; i < n_local; ++i) list_v.push_back(VID(i));
}

template<class VTYPE> void Graph<VTYPE>:: send_buf(const vector<int> &list_change, int atts_local, bool syn_all) {
	for( int i = 0; i < n_procs; ++i ) {
		bm[i].set_cache_all( false );
		bm[i].reset();
	}

	for(auto &u:list_change) {
		int lid = LID(u);
		for( int j = 0; j < n_procs; ++j )
			if(j != id && (arbitrary_pull || syn_all || (nb_ids[lid][j/8]&(1<<(j%8))))) {
				bm[j].write_int(u);
				v_all[u].write(bm[j],atts_local);
			}
	}

	for( int i = 0; i < n_procs; ++i ) bm[i].write_int(BUFEND);

	send_to_neighbor(atts_local);

	int u, i;
	init_all_bm();

	while(next_all_bm(u, i)) {
		v_all[u].read(bm[i],bm[i].atts);
		bm[i].next_id();
	}
}

template<class VTYPE> void Graph<VTYPE>::push(function<void(VTYPE&, VTYPE&, VTYPE*&, MetaInfo&)> f_push, const vector<int> &list_v, vector<int> &list_result) {
	for( size_t i = 0; i < list_v.size(); ++i ) {
		int u = list_v[i], lid = LID(u);
		set_info(u);
		f_push(v_loc[lid], v_all[u], v_all, info);
	}

	list_result.clear();
	for(int i = 0; i < (n+31)/32; ++i)
		if(bset[i]) {
			for(int j = 0; j < 32; ++j)
				if(bset[i] & (1<<j))
					list_result.push_back(i*32+j);
			bset[i] = 0;
		}

	for( int i = 0; i < n_procs; ++i ) {
		bm[i].set_cache_all( false );
		bm[i].reset();
	}

	for(size_t i = 0; i < list_result.size(); ++i) {
		int u = list_result[i];
		int cid = u%n_procs;
		bm[cid].write_int(u);
	}
	for( int i = 0; i < n_procs; ++i ) bm[i].write_int(BUFEND);

	send_to_neighbor(NONE);

	int u, i, pre = -1;
	init_all_bm(id);

	list_result.clear();
	while(next_all_bm(u, i)) {
		if( u != pre ) { list_result.push_back(u); pre = u; }
		bm[i].next_id();
	}
}


template<class VTYPE> void Graph<VTYPE>::local(function<void(VTYPE&, VTYPE*&, MetaInfo&)> f_local, const vector<int> &list_v, int atts) {
	bool synall = false;
	if(atts == -1) atts = critical_atts;
	else if(atts == SYNALL) {synall=true;atts=critical_atts;}

	vector<int> list_change;
	int atts_local = 0;
	for(auto &u:list_v) {
		set_info(u);
		v_tmp.cp_from(v_all[u],atts);
		f_local(v_all[u], v_all, info);

		int now_cmp = v_all[u].cmp(v_tmp, atts);
		if( now_cmp ) {
			list_change.push_back(u);
			atts_local |= now_cmp;
		}
	}

	if(!atts) return;
	send_buf(list_change, atts_local,synall);
}

template<class VTYPE> void Graph<VTYPE>::push(function<void(VTYPE&, VTYPE&, VTYPE*&, VTYPE*&, MetaInfo&)> f_cmb, function<void(VTYPE&, VTYPE&, VTYPE*&, MetaInfo&)> f_agg,
			const vector<int> &list_v, vector<int> &list_result, int atts_agg, int atts_cmb){
	if(v_cmb==NULL) v_cmb = new VTYPE[n];
	for( size_t i = 0; i < list_v.size(); ++i ) {
		int u = list_v[i], lid = LID(u);
		set_info(u);
		f_cmb(v_loc[lid], v_all[u], v_all, v_cmb, info);
	}

	int atts_local = 0;
	list_result.clear();
	for(int i = 0; i < (n+31)/32; ++i)
		if(bset[i]) {
			for(int j = 0; j < 32; ++j)
				if(bset[i] & (1<<j)) {
					int u = i*32+j;
					list_result.push_back(u);
					atts_local |= v_cmb[u].cmp0(atts_cmb);
				}
			bset[i] = 0;
		}

	for( int i = 0; i < n_procs; ++i ) {
		bm[i].set_cache_all( false );
		bm[i].reset();
	}

	for(auto &u:list_result) {
		int cid = u%n_procs;
		bm[cid].write_int(u);
		v_cmb[u].write(bm[cid], atts_local);
	}
	for( int i = 0; i < n_procs; ++i ) bm[i].write_int(BUFEND);

	send_to_neighbor(atts_local);

	int u, i, pre = -1;
	init_all_bm(id);

	list_result.clear();

	bool synall=false;
	if(atts_agg == -1) atts_agg = critical_atts;
	else if(atts_agg == SYNALL) {synall=true;atts_agg=critical_atts;}

	while(next_all_bm(u, i)) {
		v_tmp.init();
		v_tmp.read(bm[i],bm[i].atts);
		set_info(u);
		if( u != pre ) {
			list_result.push_back(u);
			pre = u;
			info.is_first = true;
			v_cmb[u].cp_from(v_all[u], atts_agg);
		}
		else info.is_first = false;
		f_agg(v_tmp,v_all[u],v_all,info);
		bm[i].next_id();
	}

	if(!atts_agg) return;

	atts_local = 0;
	vector<int> list_change;
	for(auto &u: list_result) {
		int lid = LID(u);
		int now_cmp = v_all[u].cmp(v_cmb[u],atts_agg);
		if(now_cmp) {
			atts_local |= now_cmp;
			list_change.push_back(u);
		}
	}
	send_buf(list_change, atts_local, synall);
}

template<class VTYPE> void Graph<VTYPE>::pull(function<void(VTYPE&, VTYPE*&, MetaInfo&)> f_pull, const vector<int> &list_v, vector<int> &list_result, int atts) {
	bool synall = false;
	if(atts == -1) atts = critical_atts;
	else if(atts == SYNALL) {synall=true;atts=critical_atts;}

	list_result.clear();
	int atts_local = 0;
	for(auto &u:list_v) {
		int lid = LID(u);
		set_info(u);
		v_loc[lid].cp_from(v_all[u],atts);
		f_pull(v_all[u], v_all, info);

		int now_cmp = v_all[u].cmp(v_loc[lid], atts);
		if( now_cmp ) {
			list_result.push_back(u);
			v_loc_tmp[lid].cp_from(v_all[u], atts);
			v_all[u].cp_from(v_loc[lid],now_cmp);
			atts_local |= now_cmp;
		}
	}

	for(auto &u:list_result) v_all[u].cp_from(v_loc_tmp[LID(u)], atts_local);
	if(!atts) return;
	send_buf(list_result, atts_local, synall);
}


template<class VTYPE> void Graph<VTYPE>::update_buf(BufManager &b, int atts) {
	if(b.cache_all) return;
	int n_bit = (n_local+7)/8 + 1;
	long long len = b.pos - b.n_element * sizeof(int) + n_bit;
	if( len * 6 >= b.pos * 5 ) return;
	b_tmp.update(len);
	memset(b_tmp.buf, 0, sizeof(char) * n_bit);
	b_tmp.pos = n_bit;
	b.pos = 0;
	for( int u = b.read_int(); u != BUFEND; u = b.read_int() ) {
		v_tmp.read(b, atts);
		v_tmp.write(b_tmp, atts);
		b_tmp.set_bit(LID(u));
	}
	b.pos = b_tmp.pos;
	b.set_cache_all( true );
	memcpy(b.buf, b_tmp.buf, b_tmp.pos);
}


template<class VTYPE> void Graph<VTYPE>::send_to_neighbor(int atts) {
	MPI_Status s_send_len, s_send_dat, s_recv_len, s_recv_dat;
	MPI_Request r_send_len, r_send_dat, r_recv_len, r_recv_dat;
	char buf_send[16], buf_recv[16];

	long long len;

	for( int i = 0; i < n_procs; ++i ) update_buf(bm[i], atts);

	for( int i = 0; i < n_procs; ++i ) {
		int dest = i;
		if( id == dest ) {bm[id].atts=atts;continue;}

		memcpy(buf_send, &bm[dest].pos, sizeof(long long));
		memcpy(buf_send + sizeof(long long), &bm[dest].cache_all, sizeof(bool));
		memcpy(buf_send + sizeof(long long) + sizeof(bool), &atts, sizeof(int) );

		MPI_Isend(buf_send, sizeof(long long) + sizeof(bool) + sizeof(int), MPI_CHAR, dest, 0, MPI_COMM_WORLD, &r_send_len);
		MPI_Irecv(buf_recv, sizeof(long long) + sizeof(bool) + sizeof(int), MPI_CHAR, dest, 0, MPI_COMM_WORLD, &r_recv_len);

		MPI_Wait(&r_send_len, &s_send_len);
		MPI_Wait(&r_recv_len, &s_recv_len);

		memcpy(&len, buf_recv, sizeof(long long));
		b_tmp.update(len);

		MPI_Isend(bm[dest].buf, bm[dest].pos, MPI_CHAR, dest, 1, MPI_COMM_WORLD, &r_send_dat);
		MPI_Irecv(b_tmp.buf, len, MPI_CHAR, dest, 1, MPI_COMM_WORLD, &r_recv_dat);

		MPI_Wait(&r_send_dat, &s_send_dat);
		MPI_Wait(&r_recv_dat, &s_recv_dat);

		bm[dest].update(len);
		memcpy(bm[dest].buf, b_tmp.buf, len);
		memcpy(&bm[dest].cache_all, buf_recv + sizeof(long long), sizeof(bool));
		memcpy(&bm[dest].atts, buf_recv + sizeof(long long) + sizeof(bool), sizeof(int));
	}
}

template<class VTYPE> void Graph<VTYPE>::init_all_bm(int start_id) {
	for( int i = 0; i < n_procs; ++i ) bm[i].first_id(start_id);
}

template<class VTYPE> void Graph<VTYPE>::set_info(int u) {
	int lid = LID(u);
	info.id = u;
	info.deg = deg[lid];
	info.nb_loaded = false;
	info.nbw_loaded = false;
	info.din = din[lid];
	info.dout = info.deg - info.din;
}

template<class VTYPE> bool Graph<VTYPE>::next_all_bm(int &vid, int &cid) {
	vid = -1; cid = -1;
	for( int i = 0; i < n_procs; ++i )
		if( !bm[i].end() )
			if( vid == -1 || bm[i].get_id() < vid ) {
				vid = bm[i].get_id();
				cid = i;
			}
	return (cid != -1);
}

template<class VTYPE> void Graph<VTYPE>::gather(function<void(VTYPE&, VTYPE*&, MetaInfo&)> func, const vector<int> &list_v, int atts) {
	bm[0].set_cache_all(false);
	bm[0].reset();

	for( size_t i = 0; i < list_v.size(); ++i ){
		int u = list_v[i];
		bm[0].write_int(u);
		v_all[u].write(bm[0],atts);
	}
	bm[0].write_int(BUFEND);

	if( !is_master ) {
		MPI_Send(&bm[0].pos, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
		MPI_Send(bm[0].buf, bm[0].pos, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	} else {
		long long len;
		MPI_Status recv_len, recv_dat;
		for( int i = 1; i < n_procs; ++i ) {
			MPI_Recv(&len, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &recv_len);
			bm[i].update(len); bm[i].set_cache_all(false); bm[i].reset();
			MPI_Recv(bm[i].buf, len, MPI_CHAR, i, 1, MPI_COMM_WORLD, &recv_dat);
		}

		int u, i;
		init_all_bm();

		while(next_all_bm(u, i)) {
			v_all[u].read(bm[i],atts);
			bm[i].next_id();
		}
		for(u=0;u<n;++u) {
			set_info(u);
			func(v_all[u], v_all, info);
		}
	}
}

template<class VTYPE> void Graph<VTYPE>::traverse(function<void(VTYPE&, VTYPE*&, MetaInfo&)> func){
	for(int u = 0; u < n; ++u) {
		info.id = u;
		func(v_all[u], v_all, info);
	}
}

#endif
