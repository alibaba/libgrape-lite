#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,f,int,deg, int, id, int, pre, ONE+TWO+THREE);
	SetDataset(argv[1], argv[2]);
	SetSynAll(true);

	vector<int> x(n_vertex), y(n_vertex), ds(n_vertex), degs(n_vertex);
	vector<vector<int>> sub(n_vertex);
	int p = 0;

	function<void(int,int)> dfs=[&](int u, int nowd) {
		x[u] = p++;
		ds[u] = nowd;
		for(auto &v:sub[u]) dfs(v,nowd+1);
		y[u] = p++;
	};

	DefineMapV(init) {v.f = -1; v.id = v_id; v.deg = v_deg;};
	vertexSubset A = vertexMap(All, CTrueV, init);

	Traverse(degs[v_id]=v.deg);
	for(int len=n_vertex, i = 0; len > 0; len = A.size(), ++i) {
		p = 0;
		Traverse(if(v.f==-1) dfs(v_id,0));

		DefineMapV(local) {v.pre = v.f;};
		A = vertexMap(A, CTrueV, local);
		DefineFE(check) {return y[s.id]<x[d.id] && (d.f == d.pre || degs[s.id]<degs[d.f]);};
		DefineMapE(update) {d.f = s.id;};
		A = edgeMapDense(All, ED, check, update, CTrueV);

		Traverse(if(v.f != -1 && y[v.f]<x[v_id]) sub[v.f].push_back(v_id));
		Traverse(int s = 0;
			for(auto u:sub[v_id])
				if(get_v(u).f == v_id)
					sub[v_id][s++] = u;
			sub[v_id].resize(s);
		);
		int maxd=0; for(auto nowd:ds) maxd=max(maxd,nowd);
		print( "Round %d: len=%d, maxd=%d\n", i, len, maxd );
	}

	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
