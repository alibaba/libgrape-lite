#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(float,val, float, next, ONE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.val = 1.0/n_vertex;};
	vertexMap(All, CTrueV, init);

	float s=0, s_all=0;
	DefineMapE(update) {d.next += s.val;};
	DefineMapV(local1) {v.val = v.next; v.next = 0; s+= v.val * v.val;};
	DefineMapV(local2) {v.val /= s_all;};

	for(int i = 0; i < 10; ++i) {
		print("Round %d\n", i);
		s=0, s_all=0;
		edgeMapDense(All, EU, CTrueE, update, CTrueV);
		vertexMap(All, CTrueV, local1);
		s_all = sqrt(Sum(s));
		vertexMap(All, CTrueV, local2);
	}

	//All.Gather(printf( "v=%d,val=%0.5f\n", v_id, v.val));
	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
