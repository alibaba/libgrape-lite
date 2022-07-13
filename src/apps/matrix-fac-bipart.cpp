#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(vector<float>,val);
	SetDataset(argv[1], argv[2]);
	int d = atoi(argv[3]);
	float alpha=0.0002, beta=0.02;

	DefineMapV(init) {
		v.val.resize(d);
		for(int i=0;i<d;++i) v.val[i]= rand()*1.0/RAND_MAX;
		return v;
	};
	vertexMap(All, CTrueV, init);
	
	DefineMapV(update) {
		for_nb(float w=nb_w-prod(v.val,nb.val); mult(v.val,1-alpha*beta); add(v.val,nb.val,2.0f*alpha*w));
		return v;
	};

	for(int i = 0; i < 500; ++i) {
		vertexMap(All, CTrueV, update);
	}

	print( "total time=%0.3lf secs\n", GetTime());
	//All.Gather(if (v_id < 10) cout<<"id="<<v_id<<",val="<<v.val<<"\n");
	return 0;
}
