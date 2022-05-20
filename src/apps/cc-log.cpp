#include "../core/api.h"
#define jump(A) edgeMapDense(All, EjoinV(edgesj, A), CTrueE, updatej, CTrueV)

#define star(A) S = vertexMap(A, CTrueV, locals); \
S = edgeMapDense(All, EjoinV(edges, S), checks1, updates, CTrueV);\
edgeMapDense(All, EjoinV(edges, S), CTrueE, updates2, CTrueV);\
edgeMapSparse(S, edges2, CTrueE, updates, CTrueV, updates);\
edgeMapDense(All, EjoinV(edges, A), checks2, updates, CTrueV);
		
#define hook(A) S = vertexMap(A, filterh1, localh1); \
edgeMapDense(All, EjoinV(EU, S), checks1, h1, CTrueV);\
edgeMapSparse(S, edges, checkh, h2, CTrueV, h2);\
vertexMap(S, filterh2, localh2);


int main(int argc, char *argv[]) {
	VertexType(int,id, int,p, bool,s, int,f, int,pp, ONE+TWO+THREE);
	SetDataset(argv[1], argv[2]);
	SetSynAll(true);
	vertexSubset S;
	bool c = true;

	DefineMapV(init) {v.id = id(v); v.p = id(v); v.s = false; v.f = id(v); return v;};
	vertexMap(All, CTrueV, init);

	DefineMapE(update1) {d.p = min(d.p, s.p); return d;};
	vertexSubset A = edgeMapDense(All, EU, CTrueE, update1, CTrueV);

	DefineOutEdges(edges) {VjoinP(p)};
	DefineMapE(update2) {d.s = true; return d;};
	edgeMapSparse(A, edges, CTrueE, update2, CTrueV, update2);

	DefineFV(filter1) {return v.p == id(v) && (!v.s);};
	DefineMapV(local1) {v.p = INT_MAX; return v;};
	A = vertexMap(All, filter1, local1);
	edgeMapDense(All, EjoinV(EU, A), CTrueE, update1, CTrueV);

	DefineFV(filter2) {return v.p != INT_MAX;};
	A = vertexMap(All, filter2);

	DefineMapE(updatej) {d.p = s.p; return d;};
	DefineInEdges(edgesj) {VjoinP(p)};

	DefineOutEdges(edges2) {VjoinP(pp)};
	DefineMapV(locals) {v.s = true; return v;};
	DefineFE(checks1) {return s.p != d.p;};
	DefineMapE(updates) {d.s = false; return d;};
	DefineMapE(updates2) {d.pp = s.p; return d;};
	DefineFE(checks2) {return s.s == false && d.s == true;};

	DefineFV(filterh1) {return v.s;};
	DefineMapE(h1) {d.f = min(d.f, s.p); return d;};
	DefineMapV(localh1) {if (c) v.f = v.p; else v.f = INT_MAX; return v;};
	DefineFE(checkh) {return s.p != s.id && s.f != INT_MAX && s.f != s.p;};
	DefineMapE(h2) {d.f = min(d.f, s.f); return d;};
	DefineFV(filterh2) {return v.p == v.id && v.f != INT_MAX && v.f != v.p;};
	DefineMapV(localh2) {v.p = v.f; return v;};


	for(int i=0, len=0; Size(A) > 0; ++i) {
		len = Size(jump(A));
		if (len == 0) break;
		print("Round %d,len=%d\n", i,len);
		jump(A); jump(A);
		star(A); c = true; hook(A);
		star(A); c = false; hook(A);
	}

	DefineFV(filter3) {return v.p == INT_MAX;};
	DefineMapV(local3) {v.p = id(v); return v;};
	vertexMap(All, filter3, local3);

	double t = GetTime();
	vector<int> cnt(n_vertex,0);
	int nc = 0, lc = 0;
	All.Gather(if( cnt[v.p] == 0 ) ++nc; ++cnt[v.p]; lc = max(lc, cnt[v.p]));
	print( "num_cc=%d, max_cc=%d\ntotal time=%0.3lf secs\n", nc, lc, t);

	return 0;
}
