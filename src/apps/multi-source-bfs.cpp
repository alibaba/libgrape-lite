#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(long long,seen,vector<char>,d,ONE);
	SetDataset(argv[1], argv[2]);

	vector<int> s; long long one = 1;
	for(int i = 3; i < argc; ++i) s.push_back(atoi(argv[i])); int l = (int) s.size();

	vertexSubset C = All;

	DefineMapV(init) {
		v.d.resize(l); 
		for (int i = 0; i < l; i++) v.d[i] = -1;
		v.seen=0;
		return v;
	};
	vertexSubset S = vertexMap(All, CTrueV, init);

	DefineFV(filter) {return find(s, id(v));};
	DefineMapV(local) {
		int p=locate(s,v_id); v.seen|=one<<p; v.d[p]=0;
		return v;
	};
	S = vertexMap(S, filter, local);

	for(int len = Size(S), i = 1; len > 0; len = Size(S), ++i) {
		print("Round %d: size=%d\n", i, len);
		DefineMapE(update) {
			long long b=s.seen&(~d.seen);
			if(b) {d.seen|=b; for(int p=0;p<l;++p) if(b&(one<<p)) d.d[p]=i;}
		};
		S = edgeMapDense(All, ED, CTrueE, update, CTrueV);

	}

	print( "total time=%0.3lf secs\n", GetTime());
	//All.Local(cout << "id=" << v_id << ",d=" << v.d << "\n" );
	return 0;
}
