#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,deg, vector<int>,out, vector<vector<int>>,ego, ONE+TWO);
	SetDataset(argv[1], argv[2]);
	int n_itr = argc>3? atoi(argv[3]) : 1;

	DefineMapV(init) {
		v.deg = deg(v); v.ego.resize(v.deg); 
	};
	vertexMap(All, CTrueV, init);

	int r;
	DefineMapV(local) {
		v.out.clear(); 
		for_nb(
			if(nb_id%n_itr==r && (nb.deg>v_deg || (nb.deg==v_deg && nb_id > v_id))) v.out.push_back(nb_id)
		);
	};

	vector<int> p(n_vertex,-1);
	DefineMapV(update) {
		int idx=0; for_nb(p[nb_id] = idx++); idx=0;
		for_nb(for(auto u:nb.out) if(p[u]>=0) {v.ego[idx].push_back(p[u]);v.ego[p[u]].push_back(idx);} ++idx);
		idx=0; for_nb(p[nb_id] = -1; sort(v.ego[idx].begin(),v.ego[idx].end()); vector<int>(v.ego[idx]).swap(v.ego[idx]); ++idx);
	};

	for(r = 0; r < n_itr; ++r) {
		print( "Round %d, Loading...\n", r+1 );
		vertexMap(All, CTrueV, local);
		print( "Computing...\n" );
		vertexMap(All, CTrueV, update, NONE);
	}

    print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
