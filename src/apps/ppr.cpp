#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(float,val,int,deg, float, next, ONE+TWO);
	SetDataset(argv[1], argv[2]);
	int s = atoi(argv[3]);


	DefineMapV(init) {v.deg = deg(v); v.next = (v_id==s?0.5:0);};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineFV(filter) {return id(v) == s;};
	DefineMapV(local) {v.val = 1.0f;};

	A = vertexMap(A, filter, local);
	DefineMapE(none) {};
	vertexSubset B = edgeMapSparse(A, EU, CTrueE, none, CTrueV, none);
	A = Union(B, A);

	for(int len = A.size(), i = 0; len > 0 && i < 20; ++i, len = A.size()) {
		print("Round %d: size=%d\n", i, len);
		DefineMapE(update) {d.next += 0.5*s.val/s.deg;};
		A = edgeMapDense(All, EU, CTrueE, update, CTrueV);

		DefineMapV(local2) {v.val = v.next; v.next = (v_id==s?0.5:0);};
		A = vertexMap(All, CTrueV, local2);
	}

	float max_val = -1; double tt = 0; double t = GetTime();
	All.Gather( if(v.val > max_val) max_val=v.val; tt += v.val);

	print( "max_val=%0.5f, t_val=%0.5lf\ntotal time=%0.3lf secs\n", max_val, tt, t);
	return 0;
}
