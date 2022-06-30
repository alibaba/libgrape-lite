#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,d, int,c);
	SetDataset(argv[1], argv[2]);
	int k = atoi(argv[3]);

	DefineMapV(init) {v.d = deg(v);};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineFV(filter) {return v.d < k;};
	DefineMapV(local) {v.c = 0; return v;};
	DefineFV(check) {return v.d >= k;};
	DefineMapE(update1) {d.c++; return d;};
	DefineMapE(update2) {d.d -= s.c; return d;};

	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print("Round %d: size=%d\n", i, len);
		A = vertexMap(A, filter, local);
		A = edgeMapSparse(A, EU, CTrueE, update1, check, update2);
	}

	double t = GetTime();
	int s = Size(vertexMap(All, check));
	print( "k-core size=%d,time=%0.3lf secs\n", s, t);
	return 0;
}
