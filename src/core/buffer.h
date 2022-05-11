#ifndef BUF_H_
#define BUF_H_

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

class BufManager {
public:
	char *buf;
	long long len;
	long long pos;
	bool cache_all;
	int atts;
	int n, n_procs, cid, now_id;
	int n_element, n_local, n_bit;

public:
	BufManager() {
		buf = new char[MAXBUF+BUFDLT]; len = MAXBUF; pos = 0; cache_all=false;
		n = 0; n_procs = 0; cid = 0; now_id = 0; n_element = 0; n_local = 0; n_bit = 0; atts = 0;
	}
	~BufManager() {delete[] buf;}

	void set_info(int n, int n_procs, int cid) {
		this->n = n;
		this->n_procs = n_procs;
		this->cid = cid;
		n_local = NLOC(cid);
		n_bit = (n_local+7)/8 + 1;
	}

	void update() { if( pos > len ) update(pos); }

	void update(int len) {
		if( len <= this->len ) return;
		while( this->len < len )
			this->len *= 2;
		char *buf_new = new char[this->len + BUFDLT];
		memcpy(buf_new, buf, pos);
		delete[] buf;
		buf = buf_new;
	}

	void set_bit(int lid) {buf[lid/8]|=(1<<(lid%8));}
	void reset() { pos = cache_all ? n_bit : 0; n_element = 0; }
	void set_cache_all(bool cache_all) {this->cache_all = cache_all;}

	//void reset(long long len) { pos = 0; if( len > this->len ) update(len); }

	int read_int() {int val; memcpy(&val, buf+pos, sizeof(int)); pos += sizeof(int); return val;}
	void write_int(int val) {memcpy(buf+pos,&val,sizeof(int)); pos += sizeof(int);  ++n_element; update(); }
	void get_next_id() {
		for(now_id += n_procs; now_id<n; now_id += n_procs) {
			int lid = LID(now_id);
			if( buf[lid/8]&(1<<(lid%8)) ) break;
		}
	}
	int first_id(int start_id=-1) {
		pos = cache_all ? n_bit : 0;
		now_id = (start_id==-1?cid:start_id) - n_procs;
		if(cache_all) get_next_id();
		else now_id = read_int();
		return now_id;
	}
	inline int get_id() {return now_id;}
	int next_id() {if(cache_all) get_next_id(); else now_id=read_int(); return now_id;}
	inline bool end() {return cache_all ? now_id>=n : now_id==BUFEND;}
};

class MetaInfo {
public:
	int n;
	int id;
	int deg, din, dout;

	int n_procs;
	int cid;

	MyReadFile *f_dat;
	MyReadFile *f_idx;
	MyReadFile *f_w;

	int *adj;
	float *adj_w;
	unsigned *bset;

	bool nb_loaded;
	bool nbw_loaded;
	bool is_first;
	bool edge_in_mem;

	int **con;
	float **con_w;
	int *deg_all;
	int *din_all;

public:

	void load_nbw() {
		if(nbw_loaded) return;
		if(edge_in_mem) {
			int lid = id/n_procs;
			adj_w = con_w[lid];
		} else GFS::load_w(*f_w, *f_idx, id/n_procs, adj_w, true);
		nbw_loaded = true;
	}

	void load_nbr() {
		if(nb_loaded) return;
		if(edge_in_mem) {
			int lid = id/n_procs;
			adj = con[lid];
			deg = deg_all[lid];
			din = din_all[lid];
		} else deg = GFS::load_nbr(*f_dat, *f_idx, id/n_procs, adj, din, true);
		dout = deg-din;
		nb_loaded=true;
	}
	int get_nbr_id(int i) {load_nbr(); return adj[i]&ALL;}
	float get_nbr_w(int i) {load_nbw(); return adj_w[i];}
};

#endif