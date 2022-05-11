#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(bool,d,long long,r,bool,b,ONE+TWO);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.d = false; v.r = deg(v) * (long long) n_vertex + id(v);};

	DefineMapV(local) {v.b = true; return v;};

	DefineFE(check) {return !s.d && s.r < d.r;};
	DefineMapE(update) {d.b = false;};
	DefineFV(filter) {return v.b;};

	DefineMapE(update2) {return d;};
	DefineFV(cond) {return !v.d;};
	DefineReduce(reduce) {d.d = true; return d;};

	DefineFV(filter2) {return !v.b;};


	vertexSubset A = vertexMap(All, CTrueV, init);
	for(int i=0, len=Size(A); len>0; ++i) {
		A = vertexMap(A, CTrueV, local);
		edgeMapDense(All, EjoinV(EU, A), check, update, filter);

		vertexSubset B = vertexMap(A, filter);
		vertexSubset C = edgeMapSparse(B, EU, CTrueE, update2, cond, reduce);
		A = Minus(A, C);
		A = vertexMap(A, filter2);
		
		int num = Size(B); len = Size(A);
		print("Round %d: size=%d, selected=%d\n", i, len, num);
	}

	int n_mis = Size(vertexMap(All, filter));
	print( "size of max independent set=%d\ntotal time=%0.3lf secs\n", n_mis, GetTime());
	return 0;
}
