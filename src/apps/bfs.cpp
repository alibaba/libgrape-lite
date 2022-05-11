#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(char,dis);
	SetDataset(argv[1], argv[2]);
	int s = atoi(argv[3]);

	DefineMapV(init) {
		v.dis = (id(v) == s) ? 0 : -1;
		return v;
	};
	
	DefineFV(filter) {return id(v) == s;};
	
	DefineCond(cond) {return v.dis == -1;};

	//DefineFE(check) {return (s.dis != -1);};

	DefineMapE(update) {d.dis = s.dis +1; return d;};

	DefineReduce(reduce) {d = s; return d;};
	
	
	vertexSubset A = vertexMap(All, CTrueV, init);
	A = vertexMap(A, filter);

	for(int len = Size(A), i = 1; len > 0; len = Size(A), ++i) {
		print("Round %d: size=%d\n", i, len);
		A = edgeMap(A, ED, CTrueE, update, cond, reduce);
	}

	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
