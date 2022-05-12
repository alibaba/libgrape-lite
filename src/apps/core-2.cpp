#include "../core/api.h"
#include <set>

int main(int argc, char *argv[]) {
	VertexType(short,core, short,old, ONE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.core = min(SHRT_MAX,deg(v));};

	DefineMapE(none) {return d;};

	vector<int> cnt(SHRT_MAX);
	DefineMapV(local) {
		v.old = v.core;
		int nowcnt = 0;
		for_nb(if(nb.core>=v.core) ++nowcnt);
		if (nowcnt >= v.core) return;
		memset(cnt.data(),0,sizeof(int)*(v.core+1));
		for_nb(++cnt[min(v.core,nb.core)]);
		for(int s=0; s+cnt[v.core]<v.core; --v.core) s += cnt[v.core];
	};

	DefineFV(filter) {return v.core != v.old;};

	vertexSubset A = vertexMap(All, CTrueV, init);
	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print( "Round %d, len=%d\n", i, len );
		if (len < THRESHOLD) 
			A = edgeMapSparse(A, EU, CTrueE, none, CTrueV, none);
		else
			A = All;
		A = vertexMap(A, CTrueV, local);
		A = vertexMap(A, filter);
	}

	long long sum_core = 0; double t = GetTime();
	All.Gather(sum_core += v.core);
	print( "sum_core=%lld\ntotal time=%0.3lf secs\n", sum_core, t);
	return 0;
}
