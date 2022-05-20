#ifndef FUNC_H_
#define FUNC_H_

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

#include "vset.h"

using namespace std;

#define id(v) v_id
#define deg(v) v_deg
#define weight nb_w
#define vertexSubset VSet
#define CTrueV cTrueV<VTYPE>
#define CTrueE cTrueE<VTYPE>
#define Union(A, B) A.Union(B)
#define Minus(A, B) A.Minus(B)
#define Intersect(A, B) A.Intersect(B)
#define EjoinV(E, V) E, V
#define VjoinP(property) vector<int> res; res.push_back(v.property); return res;

#define Size(U) U.size()
#define edgeMap(U, H, F, M, C, R) edgeMapFunction(G, U, H, F, M, C, R)
#define edgeMapDense(U, H, F, M, C, ...) edgeMapDenseFunction(G, U, H, F, M, C, ##__VA_ARGS__)
#define edgeMapSparse(U, H, F, M, C, R) edgeMapSparseFunction(G, U, H, F, M, C, R)

#define DefineFV(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_f_v(F) F(v, v_all, info)

#define DefineCond(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)->bool
#define use_cond(F) F(nb, v_all, info)

#define DefineMapV(F) auto F=[&](VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_map_v(F) F(v,v_all,info)

#define DefineFE(F) auto F=[&](VTYPE& s, VTYPE& d, VTYPE*& v_all, MetaInfo& info)->bool
#define use_f_dense(F) F(nb, v, v_all,info)
#define use_f_sparse(F) F(v, nb, v_all,info)

#define DefineMapE(F) auto F=[&](VTYPE& s, VTYPE& d, VTYPE*& v_all, MetaInfo& info)
#define DefineDense(F) auto F=[&](VTYPE& nb, VTYPE& v, VTYPE*& v_all, MetaInfo& info)
#define use_dense(F) F(nb, v, v_all,info)
#define DefineSparse(F) auto F=[&](VTYPE& v, VTYPE& nb, VTYPE*& v_all, MetaInfo& info)
#define use_sparse(F) F(v, _v, v_all,info)

#define DefineReduce(F) auto F=[&](VTYPE& s, VTYPE& d, VTYPE*& v_all, MetaInfo& info)
#define use_reduce(F) F(_v, dst, v_all,info)

#define DefineOutEdges(F) auto F=[&](VTYPE& v) -> vector<int>
#define DefineInEdges(F) auto F=[&](VTYPE& v) -> vector<int>
#define use_edge(F) F(v)

template<class VTYPE>
inline bool cTrueV(VTYPE& v, VTYPE*& v_all, MetaInfo& info) {
	return true;
}

template<class VTYPE>
inline bool cTrueE(VTYPE& s, VTYPE& d, VTYPE*& v_all, MetaInfo& info) {
	return true;
}

#endif