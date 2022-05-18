#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,fid, int,scc);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.scc = -1; return v;};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineMapV(local1) {v.fid = id(v); return v;};

	DefineFV(filter2) {return v.fid == id(v);};
	DefineMapV(local2) {v.scc = id(v); return v;};

	DefineFV(filter3) {return v.scc == -1;};
	
	for(int i = 1, na = Size(A); na>0 ;na = Size(A), ++i) {
		vertexSubset B = vertexMap(A, CTrueV, local1);
		for(int nb = Size(B), j = 1; nb>0; nb = Size(B), ++j) {
			print( "Round %d.1.%d: na=%d, nb=%d\n", i, j, na, nb );

			DefineFE(check1) {return s.fid < d.fid;};
			DefineMapE(update1) {d.fid = min(d.fid, s.fid); return d;};
			DefineFV(cond1) {return v.scc == -1;};

			B = edgeMap(B, EjoinV(ED, A), check1, update1, cond1, update1);
		}

		B = vertexMap(A, filter2, local2);
		for(int nb = Size(B), j = 1; nb>0; nb = Size(B), ++j) {
			print( "Round %d.2.%d: na=%d, nb=%d\n", i, j, na, nb );

			DefineFE(check2) {return s.scc == d.fid;};
			DefineMapE(update2) {d.scc = d.fid; return d;};
			DefineFV(cond2) {return v.scc == -1;};
			
			B = edgeMap(B, EjoinV(ER, A), check2, update2, cond2, update2);
		}
		A = vertexMap(A, filter3);
	}

	double t = GetTime();
	vector<int> cnt(n_vertex,0);
	int nc = 0, lc = 0;
	All.Gather(if(cnt[v.scc%n_vertex] == 0) ++nc; ++cnt[v.scc%n_vertex]; lc = max(lc, cnt[v.scc%n_vertex]));
	print( "num_scc=%d, max_scc=%d\ntotal time=%0.3lf secs\n", nc, lc, t);

	return 0;
}
