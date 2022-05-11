#ifndef VSET_H_
#define VSET_H_

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

#include "graph.h"

using namespace std;

#define FUNC_FILTER(F) [&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)->bool{return F;}
#define FUNC_PULL(F) [&](VTYPE& v, VTYPE*& v_all, MetaInfo& info){F;}
#define FUNC_LOCAL(F) [&](VTYPE& v, VTYPE*& v_all, MetaInfo& info){F;}
#define FUNC_GATHER(F) [&](VTYPE& v, VTYPE*& v_all, MetaInfo& info){F;}
#define FUNC_TRAVERSE(F) [&](VTYPE& v, VTYPE*& v_all, MetaInfo& info){F;}
#define FUNC_PUSH(F) [&](VTYPE& old, VTYPE& v, VTYPE*& v_all,  MetaInfo& info){F;}
#define FUNC_CMB(F) [&](VTYPE& old, VTYPE& v, VTYPE*& v_all, VTYPE*& v_cmb, MetaInfo& info){F;}
#define FUNC_AGG(F) [&](VTYPE& _v, VTYPE& dst, VTYPE*& v_all, MetaInfo& info){F;}
#define FUNC_BLOCK(F) [&](){F;}

#define Pull(F, ...) pull(FUNC_PULL(F), ##__VA_ARGS__)
#define Local(F, ...) local(FUNC_LOCAL(F), ##__VA_ARGS__)
#define Gather(F, ...) gather(FUNC_GATHER(F), ##__VA_ARGS__)
#define Traverse(F, ...) All.traverse(FUNC_TRAVERSE(F), ##__VA_ARGS__)
#define Filter(F, ...) filter(FUNC_FILTER(F), ##__VA_ARGS__)
#define Push1(F) push(FUNC_PUSH(F))
#define Push2(F1,F2, ...) push(FUNC_CMB(F1),FUNC_AGG(F2), ##__VA_ARGS__)
#define Block(F) block(FUNC_BLOCK(F))

#define DefineFilter(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_filter(F) F(v, v_all, info)

#define DefinePull(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_pull(F) F(v,v_all,info)

#define DefinePush(F) auto F=[&](VTYPE& old, VTYPE& v, VTYPE*& v_all,  MetaInfo& info)
#define use_push(F) F(old,v,v_all,info)

#define DefineCmb(F) auto F=[&](VTYPE& old, VTYPE& v, VTYPE*& v_all, VTYPE*& v_cmb, MetaInfo& info)
#define use_cmb(F) F(old,v,v_all,v_cmb,info)

#define DefineAgg(F) auto F=[&](VTYPE& _v, VTYPE& dst, VTYPE*& v_all, MetaInfo& info)
#define use_agg(F) F(_v,dst,v_all,info)

#define DefineLocal(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_local(F) F(v,v_all,info)

#define GetPush(_0, _1, _2, _3, _4, NAME, ...) NAME
#define Push(...) GetPush(_0, ##__VA_ARGS__, Push2, Push2, Push2, Push1, push, ...)(__VA_ARGS__)

#define push_to_1(ID) {int myid=ID; info.bset[myid/32] |= (1<<(myid%32));}
#define push_to_2(ID,F)  {int myid=ID; VTYPE& _v=v_cmb[myid]; if(!(info.bset[myid/32]&(1<<(myid%32)))) { _v.init();info.bset[myid/32] |= (1<<(myid%32));} F;}
#define push_to_3(ID,INIT,F) {int myid=ID; VTYPE& _v=v_cmb[myid]; if(!(info.bset[myid/32]&(1<<(myid%32)))) { _v.init();INIT;info.bset[myid/32] |= (1<<(myid%32));}  F;}
#define reduce(INIT,F) {if(info.is_first) {INIT;} F;}

#define get_push_to(_1, _2, _3, NAME, ...) NAME
#define push_to(...) get_push_to(__VA_ARGS__, push_to_3, push_to_2, push_to_1, ...) (__VA_ARGS__)

#define All VertexSet<VTYPE>::all
#define Empty VertexSet<VTYPE>::empty
#define VSet VertexSet<VTYPE>

template<class T> class VertexSet {
public:
	static VertexSet<T> all;
	static VertexSet<T> empty;
	static Graph<T> *g;

public:
	vector<int> s;
	unsigned *bitset = NULL;
	int size();
	bool is_empty();

	VertexSet<T> Union(const VertexSet<T> &x);
	VertexSet<T> Minus(const VertexSet<T> &x);
	VertexSet<T> Intersect(const VertexSet<T> &x);
	VertexSet<T> pull(function<void(T&, T*&, MetaInfo&)> f_pull, int atts=-1);
	VertexSet<T> push(function<void(T&, T&, T*&, MetaInfo&)> f_push);
	VertexSet<T> push(function<void(T&, T&, T*&, T*&, MetaInfo&)> f_cmb, function<void(T&, T&, T*&, MetaInfo&)> f_agg, int atts_agg=-1, int atts_cmb=-1);


	VertexSet<T> &local(function<void(T&, T*&, MetaInfo&)> f_local, int atts=-1);
	VertexSet<T> &gather(function<void(T&, T*&, MetaInfo& info)> func, int atts=-1);
	VertexSet<T> filter(function<bool(T&, T*&, MetaInfo&)> f_filter);
	VertexSet<T> &block(function<void()> f);
	void traverse(function<void(T&, T*&, MetaInfo& info)> func);
	void sync();
	bool contain(int u);
};

//implementation


template<class T> VertexSet<T> VertexSet<T>::all = VertexSet<T>();
template<class T> VertexSet<T> VertexSet<T>::empty = VertexSet<T>();
template<class T> Graph<T>* VertexSet<T>::g = NULL;

template<class T> bool VertexSet<T>::is_empty() {
	return size() == 0;
}

template<class T> int VertexSet<T>::size() {
	int cnt_local = (int) s.size(), cnt;
	MPI_Allreduce(&cnt_local, &cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	return cnt;
}

template<class T>  VertexSet<T> VertexSet<T>::pull(function<void(T&, T*&, MetaInfo&)> f_pull, int atts) {
	VertexSet<T> x;
	g->pull(f_pull, s, x.s, atts);
	return x;
}

template<class T> VertexSet<T> VertexSet<T>::push(function<void(T&, T&, T*&, MetaInfo&)> f_push) {
	VertexSet<T> x;
	g->push(f_push, s, x.s);
	return x;
}

template<class T> VertexSet<T> VertexSet<T>::push(function<void(T&, T&, T*&, T*&, MetaInfo&)> f_cmb,
		function<void(T&, T&, T*&, MetaInfo&)> f_agg, int atts_agg, int atts_cmb) {
	VertexSet<T> x;
	g->push(f_cmb, f_agg, s, x.s, atts_agg, atts_cmb);
	return x;
}


template<class T> VertexSet<T>& VertexSet<T>::local(function<void(T&, T*&, MetaInfo&)> f_local, int atts) {
	VertexSet<T> x;
	g->local(f_local, s, atts);
	return *this;
}

template<class T> VertexSet<T>& VertexSet<T>::block(function<void()> f) {
	f();
	return *this;
}

template<class T> VertexSet<T>& VertexSet<T>::gather(function<void(T&, T*&, MetaInfo& info)> func, int atts) {
	g->gather(func, s, atts);
	return *this;
}

template<class T> void VertexSet<T>::traverse(function<void(T&, T*&, MetaInfo& info)> func) {
	g->traverse(func);
}

template<class T> VertexSet<T> VertexSet<T>::filter(function<bool(T&, T*&, MetaInfo&)> f_filter) {
	VertexSet<T> x;
	g->filter(f_filter, s, x.s);
	return x;
}

template<class T>  VertexSet<T> VertexSet<T>::Union(const VertexSet<T> &x) {
	VertexSet y;
	y.s.resize(s.size() + x.s.size());
	vector<int>::iterator it = set_union(s.begin(), s.end(), x.s.begin(), x.s.end(), y.s.begin());
	y.s.resize(it-y.s.begin());
	return y;
}

template<class T>  VertexSet<T> VertexSet<T>::Minus(const VertexSet<T> &x) {
	VertexSet y;
	y.s.resize(s.size());
	vector<int>::iterator it = set_difference(s.begin(), s.end(), x.s.begin(), x.s.end(), y.s.begin());
	y.s.resize(it-y.s.begin());
	return y;
}

template<class T> VertexSet<T> VertexSet<T>::Intersect(const VertexSet<T> &x) {
	VertexSet y;
	y.s.resize(min(s.size(),x.s.size()));
	vector<int>::iterator it = set_intersection(s.begin(), s.end(), x.s.begin(), x.s.end(), y.s.begin());
	y.s.resize(it-y.s.begin());
	return y;
}

template<class T> void VertexSet<T>::sync() {
	//cout<<"sync"<<endl;
	int set_size = (g->n+31)/32;
	if (bitset == NULL) {
		bitset = new unsigned[set_size];
	}
	unsigned *tmp = new unsigned[set_size];
	memset(tmp, 0, sizeof(unsigned) * set_size);
	for(auto &u:s) {
		tmp[u/32] |= (1 << (u%32));
	}
	MPI_Reduce(tmp, bitset, set_size, MPI_UNSIGNED, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_Bcast(bitset, set_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	return;
}

template<class T> bool VertexSet<T>::contain(int u) {
	return (bitset[u/32] & (1 << (u%32)));
}

#endif
