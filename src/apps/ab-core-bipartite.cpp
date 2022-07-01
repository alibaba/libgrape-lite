#include "../core/api.h"
#define C(ID,V) ((ID<n_x && V.d<ta) || (ID>=n_x && V.d<tb))

int main(int argc, char *argv[]) {
	VertexType(int,d, int, id, int,c);
	SetDataset(argv[1], argv[2]);
	int ta = atoi(argv[3]), tb = atoi(argv[4]);

	DefineMapV(init) {v.id = id(v); v.d = deg(v); v.c = 0;};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineMapV(local) {v.c = 0;};
	DefineFV(check) {return C(v.id, v);};
	DefineFV(check2) {return !C(v.id, v);};

	DefineMapE(update) {d.c++;};
	DefineMapE(reduce) {d.d -= s.c;};

	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print("Round %d: size=%d\n", i, len);
		A = vertexMap(A, CTrueV, local);
		A = vertexMap(A, check);
		A = edgeMapSparse(A, EU, CTrueE, update, check2, reduce);
	}
	double t = GetTime();

	DefineFV(filter1) {return id(v)<n_x && v.d>=ta;};
	DefineFV(filter2) {return id(v)>=n_x && v.d>=tb;};
	int sa = Size(vertexMap(All, filter1)); 
	int sb = Size(vertexMap(All, filter2)); 
	print( "sa=%d,sb=%d,time=%0.3lf secs\n", sa, sb, t);
	return 0;
}
