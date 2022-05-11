#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,p,int,s,int,id);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.s = -1; v.p = -1; v.id = v_id;};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineMapV(local1) {v.p = -1;};
	DefineFE(check1) {return s.s == -1;};
	DefineMapE(update1) {d.p = max(d.p, s.id);};
	DefineCond(cond) {return v.s == -1;};

	DefineOutEdges(edges) {VjoinP(p)};
	DefineFE(check2) {return s.p != -1 && d.p == s.id;};
	DefineMapE(update2) {d.s = d.p; return d;};

	DefineFE(check3) {return d.p == s.id;};
	DefineMapE(update3) {return d;};

	for(int i=0, len=A.size(); len>0; ++i, len=A.size()) {
		print("Round %d: size=%d\n", i, len);

		A = vertexMap(A, CTrueV, local1);
		A = edgeMapDense(All, EjoinV(EU, A), check1, update1, cond);
		A = edgeMapSparse(A, edges, check2, update2, cond, update2);
		vertexSubset B = edgeMapSparse(A, edges, check2, update2, cond, update2);

		A = Union(A, B);
		A = edgeMapSparse(A, EU, check3, update3, cond, update3);
	}

	DefineFV(filter) {return v.s >= 0;};
	int n_match = Size(vertexMap(All, filter))/2;
	print( "number of matching pairs=%d\ntotal time=%0.3lf secs\n", n_match, GetTime());
	return 0;
}
