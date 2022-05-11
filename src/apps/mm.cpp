#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,p,int,s, int, id);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.s = -1; v.id = v_id;};

	DefineMapV(local) {v.p = -1;};

	DefineFV(cond) {return v.s == -1;};
	//DefineFE(check1) {return s.s == -1;};
	DefineMapE(update1) {d.p = max(d.p, s.id);};
	DefineReduce(merge1) {d.p = max(d.p, s.p);};

	DefineFE(check2) {return s.p == d.id && d.p == s.id;};
	DefineMapE(update2) {d.s = s.id;};
	DefineReduce(merge2) {d.s = s.s;};

	DefineFV(filter) {return v.s >= 0;};

	vertexSubset A = vertexMap(All, CTrueV, init);

	for(int i=0, len=Size(A); len>0; ++i, len=Size(A)) {
		print("Round %d: size=%d\n", i, len);
		A = vertexMap(A, CTrueV, local);

		A = edgeMap(A, EjoinV(EU, A), CTrueE, update1, cond, merge1);
		edgeMap(A, EjoinV(EU, A), check2, update2, cond, merge2);

		A = vertexMap(A, cond);
	}

	int n_match = Size(vertexMap(All, filter))/2;
	print( "number of matching pairs=%d\ntotal time=%0.3lf secs\n", n_match, GetTime());
	return 0;
}
