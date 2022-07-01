#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(float,val,float,next,int,dout);
	SetDataset(argv[1], argv[2]);

	float a=0, avg=0;

	DefineMapV(init) {a+=v_dout; v.dout=v_dout;};
	vertexMap(All, CTrueV, init);
	avg = Sum(a)/n_vertex;

	DefineMapE(update) {d.next += 0.85*s.val/(s.dout+avg);};
	DefineMapV(local) {v.val = 0.15+v.next; v.next = 0;};

	for(int i = 0; i < 10; ++i) {
		print("Round %d\n", i);
		edgeMapDense(All, ED, CTrueE, update, CTrueV);
		vertexMap(All, CTrueV, local);
	}

    float max_val = -1; double tt = 0, t = GetTime();
	All.Gather( if(v.val > max_val) max_val=v.val; tt += v.val);

	print( "max_val=%0.5f, t_val=%0.5lf\ntotal time=%0.3lf secs\n", max_val, tt, t);

	return 0;
}


