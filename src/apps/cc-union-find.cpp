#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,id);
	SetDataset(argv[1], argv[2]);

	union_find f(n_vertex), cc;

	DefineMapV(init) {v.id = id(v);};
	vertexSubset A = vertexMap(All, CTrueV, init);

	DefineFE(check) {return s.id > d.id;};
	DefineMapE(update) {union_f(f, s.id, d.id);};
	A = edgeMapDense(All, EU, check, update, CTrueV);

	//A.Pull(for_nb(if(nb_id>v_id) union_f(f,nb_id,v_id)));
	Reduce(f,cc,for_i(union_f(cc,f[i],i)));
	print( "Total time=%0.3lf secs\n", GetTime());

	vector<int> cnt(n_vertex,0);
	int nc = 0, lc = 0;
	for(int i = 0; i < n_vertex; ++i) {
		int fi=get_f(cc,i); if(cnt[fi] == 0 ) ++nc; ++cnt[fi]; lc = max(lc, cnt[fi]);
	}
	print( "num_cc=%d, max_cc=%d\n", nc, lc);
	return 0;
}
