#ifndef REDUCE_H_
#define REDUCE_H_

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

#include "type.h"
#include "atts.h"

template<class T> void reduce_vec(vector<T> &src, vector<T> &rst, void (*f) (void *la, void* lb, int *len, MPI_Datatype *dtype), bool bcast) {
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	bool is_master = (id==0);
	MPI_Datatype type;
	MPI_Type_contiguous(sizeof(src[0])*src.size()+sizeof(int), MPI_CHAR, &type);
	MPI_Type_commit(&type);
	MPI_Op op;
	MPI_Op_create(f, 1, &op);

	char *tmp_in = new char[sizeof(src[0])*src.size() + sizeof(int)];
	int len = (int) src.size();
	memcpy(tmp_in, &len, sizeof(int));
	memcpy(tmp_in+sizeof(int), src.data(), sizeof(src[0])*src.size());

	char *tmp_out = is_master? new char[sizeof(src[0])*src.size() + sizeof(int)] : NULL;

	MPI_Reduce(tmp_in, tmp_out, 1, type, op, 0, MPI_COMM_WORLD);
	MPI_Op_free(&op);

	delete[] tmp_in;
	if(is_master) {
		rst.resize(len);
		memcpy(rst.data(), tmp_out+sizeof(int), sizeof(src[0])*src.size());
		delete[] tmp_out;
	}

	if(bcast) {
		if(!is_master) rst.resize(src.size());
		MPI_Bcast(rst.data(), sizeof(rst[0]) * rst.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
	}
}

#define ReduceVec3(src,rst,F) reduce_vec(src, rst, [](void *la, void *lb, int *lens, MPI_Datatype *dtype) { \
	using T = decltype(src.data()); int len; memcpy(&len, la, sizeof(int)); memcpy(lb, &len, sizeof(int));\
	T src = (T) (((char*)la)+sizeof(int)); T rst = (T) (((char*)lb)+sizeof(int)); F;\
},true);

#define ReduceVec4(src,rst,F,bcast) reduce_vec(src, rst, [](void *la, void* lb, int *lens, MPI_Datatype *dtype) { \
	using T = decltype(src.data()); int len; memcpy(&len, la, sizeof(int)); memcpy(lb, &len, sizeof(int));\
	T src = (T) (((char*)la)+sizeof(int)); T rst = (T) (((char*)lb)+sizeof(int)); F;\
}, bcast);

#define GetReduceVec(_1, _2, _3, _4, NAME, ...) NAME
#define ReduceVec(...) GetReduceVec(__VA_ARGS__, ReduceVec4, ReduceVec3, _2, _1, ...) (__VA_ARGS__)
#define Reduce ReduceVec

#define ReduceVal3(src,rst,F) reduce_val(src, rst, [](void *la, void *lb, int *lens, MPI_Datatype *dtype) { \
	using T = decltype(src); T& src = *((T*)la); T& rst = *((T*)lb); F;\
},true);

#define ReduceVal4(src,rst,F,bcast) reduce_val(src, rst, [](void *la, void *lb, int *lens, MPI_Datatype *dtype) { \
	using T = decltype(src); T& src = *((T*)la); T& rst = *((T*)lb); F;\
},bcast);

#define GetReduceVal(_1, _2, _3, _4, NAME, ...) NAME
#define ReduceVal(...) GetReduceVal(__VA_ARGS__, ReduceVal4, ReduceVal3, _2, _1, ...) (__VA_ARGS__)

template<class T> void reduce_val(T &src, T &rst, void (*f) (void *la, void* lb, int *len, MPI_Datatype *dtype), bool bcast) {
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	bool is_master = (id==0);
	MPI_Datatype type;
	MPI_Type_contiguous(sizeof(src), MPI_CHAR, &type);
	MPI_Type_commit(&type);
	MPI_Op op;
	MPI_Op_create(f, 1, &op);

	MPI_Reduce(&src, &rst, 1, type, op, 0, MPI_COMM_WORLD);
	MPI_Op_free(&op);

	if(bcast) MPI_Bcast(&rst, sizeof(rst), MPI_CHAR, 0, MPI_COMM_WORLD);
}

template<class C> C Max(C src) {C rst=src; ReduceVal(src,rst,rst=max(src,rst)); return rst; }
template<class C> C Min(C src) {C rst=src; ReduceVal(src,rst,rst=min(src,rst)); return rst; }
template<class C> C Sum(C src) {C rst=0; ReduceVal(src,rst,rst+=src); return rst; }

template<class T> void Bcast(vector<T> &rst) {
	int len = rst.size();
	MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	rst.resize(len);
	MPI_Bcast(rst.data(), sizeof(rst[0]) * rst.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
}

template<class T> void Bcast(T &rst) {
	MPI_Bcast(&rst, sizeof(rst), MPI_CHAR, 0, MPI_COMM_WORLD);
}

#endif