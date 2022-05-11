#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(long long,cid);
	SetDataset(argv[1], argv[2]);

	long long v_loc = 0, v_glb = 0;
	
	DefineMapV(init) {
		v.cid=deg(v)*(long long) n_vertex + id(v); 
		v_loc=max(v_loc,v.cid);
		return v;
	};
	DefineFV(filter) {return v.cid == v_glb;};

	vertexMap(All, CTrueV, init);
	v_glb = Max(v_loc);
	vertexSubset A = vertexMap(All, filter);

	DefineFV(cond) {return v.cid != v_glb;};
	//DefineFE(check) {return s.cid == v_glb;};
	DefineMapE(update) {d.cid = v_glb; return d;};
	DefineReduce(reduce) {d = s; return d;};

	for(int len = Size(A), i = 0; len > 0; len = Size(A), ++i) {
		print("Round 0.%d: size=%d\n", i, len);
		A = edgeMap(A, EU, CTrueE, update, cond, reduce);
	}

	DefineFV(filter2) {return v.cid != v_glb;};
	A = vertexMap(All, filter2);

	DefineFE(check2) {return s.cid > d.cid;};
	DefineMapE(update2) {d.cid = max(d.cid, s.cid); return d;};

	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print("Round 1.%d: size=%d\n", i, len);
		A = edgeMap(A, EU, check2, update2, CTrueV, update2);
	}

	double t = GetTime();
	vector<int> cnt(n_vertex,0);
	int nc = 0, lc = 0;
	All.Gather(if( cnt[v.cid%n_vertex] == 0 ) ++nc; ++cnt[v.cid%n_vertex]; lc = max(lc, cnt[v.cid%n_vertex]));

	print( "num_cc=%d, max_cc=%d\ntotal time=%0.3lf secs\n", nc, lc, t);
	return 0;
}
