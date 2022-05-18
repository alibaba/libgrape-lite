#ifndef API_H_
#define API_H_

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

#include "func.h"
#include "vset.h"


using namespace std;

template<class VTYPE, class F, class M> 
inline VSet vertexMap(VSet &U, F &f, M &m, bool b = true) {
	if (b) 
		return U.Filter(use_f_v(f)).Local(use_map_v(m));
	else
		return U.Filter(use_f_v(f)).Local(use_map_v(m), NONE);

}

template<class VTYPE, class F> 
inline VSet vertexMap(VSet &U, F &f) {
	return U.Filter(use_f_v(f));
}

template<class VTYPE, class F, class M, class C, class H> 
VSet edgeMapDenseFunction(Graph<VTYPE> &G, VSet &U, H h, F &f, M &m, C &c, bool b = true) {
	return edgeMapDenseFunction(G, U, h, All, f, m, c, b);
}

template<class VTYPE, class F, class M, class C> 
VSet edgeMapDenseFunction(Graph<VTYPE> &G, VSet &U, int h, VSet &T, F &f, M &m, C &c, bool b = true) {
	bool flag = ((&U) == (&All));
	if (!flag) U.sync();
	VSet res;
	if (h == EU) {
		DefinePull(pull) {
			if (use_filter(c))
				for_nb(if (flag || U.contain(nb_id)) if (use_f_dense(f)) use_dense(m); if (!use_filter(c)) break);
		};
		if (b) res = T.Pull(use_pull(pull));
		else res = T.Pull(use_pull(pull), NONE);
	} else if (h == ED) {
		DefinePull(pull) {
			if (use_filter(c))
				for_in(if (flag || U.contain(nb_id)) if (use_f_dense(f)) use_dense(m); if (!use_filter(c)) break);
		};
		if (b) res = T.Pull(use_pull(pull));
		else res = T.Pull(use_pull(pull), NONE);
	} else if (h == ER) {
		DefinePull(pull) {
			if (use_filter(c))
				for_out(if (flag || U.contain(nb_id)) if (use_f_dense(f)) use_dense(m); if (!use_filter(c)) break);
		};
		if (b) res = T.Pull(use_pull(pull));
		else res = T.Pull(use_pull(pull), NONE);
	} else {
		res = U;
	}
	return res;
}

template<class VTYPE, class F, class M, class C, class H> 
VSet edgeMapDenseFunction(Graph<VTYPE> &G, VSet &U, H &h, VSet &T, F &f, M &m, C &c, bool b = true) {
	bool flag = ((&U) == (&All));
	if (!flag) U.sync();
	VSet res;
	DefinePull(pull) {
		auto e = use_edge(h);
		for (auto &i: e) {
			VTYPE nb = get_v(i);
			if (flag || U.contain(i))
			if (use_filter(c) && use_f_dense(f)) 
				use_dense(m); 
		}
	};
	if (b) res = T.Pull(use_pull(pull));
	else res = T.Pull(use_pull(pull), NONE);
	return res;
}

template<class VTYPE, class F, class M, class C, class R> 
VSet edgeMapSparseFunction(Graph<VTYPE> &G, VSet &U, int h, F &f, M &m, C &c, R &r) {
	VSet res;
	if (h == EU) {
		DefineCmb(cmb) {
			for_nb(if (use_cond(c) && use_f_sparse(f)) push_to(nb_id, _v = nb, use_sparse(m)));
		};
		res = U.Push(use_cmb(cmb), use_reduce(r));
	} else if (h == ED) {
		DefineCmb(cmb) {
			for_out(if (use_cond(c) && use_f_sparse(f)) push_to(nb_id, _v = nb, use_sparse(m)));
		};
		res = U.Push(use_cmb(cmb), use_reduce(r));
	} else if (h == ER) {
		DefineCmb(cmb) {
			for_in(if (use_cond(c) && use_f_sparse(f)) push_to(nb_id, _v = nb, use_sparse(m)));
		};
		res = U.Push(use_cmb(cmb), use_reduce(r));
	} else {
		res = U;
	}
	return res;
}

template<class VTYPE, class F, class M, class C, class R, class H> 
VSet edgeMapSparseFunction(Graph<VTYPE> &G, VSet &U, H &h, F &f, M &m, C &c, R &r) {
	VSet res;
	DefineCmb(cmb) {
		auto e = use_edge(h);
		for (auto &i: e) {
			VTYPE nb = get_v(i);
			if (use_cond(c) && use_f_sparse(f))
				push_to(i, _v = nb, use_sparse(m));
		}
	};
	res = U.Push(use_cmb(cmb), use_reduce(r));
	return res;
}


template<class VTYPE, class F, class M, class C, class R, class H> 
VSet edgeMapFunction(Graph<VTYPE> &G, VSet &U, H h, F &f, M &m, C &c, R &r) {
	int len = Size(U);
	if (len > THRESHOLD) 
		return edgeMapDenseFunction(G, U, h, f, m, c);
	else 
		return edgeMapSparseFunction(G, U, h, f, m, c, r);
}

template<class VTYPE, class F, class M, class C, class R, class H> 
VSet edgeMapFunction(Graph<VTYPE> &G, VSet &U, H h, VSet &T, F &f, M &m, C &c, R &r) {
	return edgeMapDenseFunction(G, U, h, T, f, m, c);
}

#endif

