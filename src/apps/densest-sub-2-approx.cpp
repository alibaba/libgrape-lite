#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(short,core, short, t, ONE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {
		v.core = min(SHRT_MAX, deg(v));
		v.t = v.core;
		return v;
	};
	vertexSubset A = vertexMap(All, CTrueV, init);

	vector<int> cnt(SHRT_MAX);
	DefineMapV(update) {
		int nowcnt = 0;
		for_nb(if(nb.core>=v.core) ++nowcnt);
		if(nowcnt >= v.core) return;
		memset(cnt.data(),0,sizeof(int)*(v.core+1));
		for_nb(++cnt[min(v.core,nb.core)]);
		for(int s=0; s+cnt[v.core]<v.core; --v.core) s += cnt[v.core];
	};

	DefineFV(filter) {
		return v.core != v.t;
	};
	DefineMapV(m) {
		v.t = v.core; return v;
	};
	for(int len = Size(A), i=0; len > 0; len = Size(A), ++i) {
		print( "Round %d, len=%d\n", i, len );
		A = vertexMap(All, CTrueV, update);
		A = vertexMap(All, filter, m);
	}

	short cloc = 0, cmax = 0;
	DefineMapV(local) {
		cloc=max(cloc,v.core);
	};
	vertexMap(All, CTrueV, local);
	cmax=Max(cloc);

	int nvloc=0,neloc=0;

	DefineFV(check) {
		return v.core == cmax;
	};
	DefineMapV(local2) {
		for_nb(if(nb.core==cmax) ++neloc);
		++nvloc;
	};
	vertexMap(All, check, local2);

	int nv=Sum(nvloc), ne=Sum(neloc);

	print( "num_vertex=%d, num_edge=%d, density=%0.5lf, cmax=%d, time=%0.3lf secs\n", nv, ne/2, ne*1.0/nv, (int)cmax, GetTime());
	return 0;
}
