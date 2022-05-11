#include "../core/api.h"
#include <set>

int main(int argc, char *argv[]) {
	VertexType(short,c,int,deg,int,id,short,cc,vector<int>,colors, ONE+TWO+THREE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.c = 0; v.deg = deg(v); v.id = id(v); v.colors.clear(); return v;};

	DefineFE(check) {return (s.deg > d.deg) || (s.deg == d.deg && s.id > d.id);};
	DefineMapE(update) {d.colors.push_back(s.c);};
	
	DefineMapV(local1) {
		set<int> used;
		used.clear();
		for (auto &i:v.colors) used.insert(i);
		for (int i=0; ; ++i) if(used.find(i) == used.end()) {v.cc=i; break;}
		v.colors.clear();
	};

	DefineFV(filter) {return v.cc != v.c;};
	DefineMapV(local2) {v.c = v.cc;};

	vertexSubset A = vertexMap(All, CTrueV, init);
	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print("Round %d: size=%d\n", i, len);
		A = edgeMapDense(All, EU, check, update, CTrueV);
		A = vertexMap(All, CTrueV, local1);
		A = vertexMap(All, filter, local2);
	}

	short max_color = 0; long long sum_color = 0; double t = GetTime();
	All.Gather(max_color=max(max_color,v.c); sum_color+=v.c);

	print( "max_color=%d, sum_color=%lld\ntotal time=%0.3lf secs\n", max_color, sum_color, t);
	return 0;
}
