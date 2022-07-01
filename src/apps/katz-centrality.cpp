#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(float,val, float, next, ONE);
	SetDataset(argv[1], argv[2]);

	float alpha = 0.1;
	DefineMapV(init) {v.val = 1.0; v.next = 0;};
	vertexMap(All, CTrueV, init);

	DefineMapE(update) {d.next += s.val+1; };
	DefineMapV(local) {v.val = v.next * alpha; v.next = 0;};

	for(int i = 0; i < 10; ++i) {
		print("Round %d\n", i);
		edgeMapDense(All, ER, CTrueE, update, CTrueV);
		vertexMap(All, CTrueV, local);
	}

	//All.Gather(printf( "v=%d,val=%0.5f\n", v_id, v.val));
	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
