#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(int,deg, int,id, vector<int>,out, long long,count, ONE+TWO+THREE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {
		v.id = id(v); v.deg = deg(v); v.count = 0; v.out.clear();
		return v;
	};

	DefineFE(check) {return (s.deg > d.deg) || (s.deg == d.deg && s.id > d.id);};
	DefineMapE(update) {d.out.push_back(s.id); return d;};

	vector<int> res(n_vertex);
	DefineMapE(update2) {
		long long p = set_intersect(s.out, d.out, res);
		d.count += (s.out.size() - 1) * (d.out.size() - 1) - p * p;
	};
	
	
	vertexMap(All, CTrueV, init);
	edgeMapDense(All, EU, CTrueE, update, CTrueV);
	edgeMapDense(All, EU, check, update2, CTrueV, false);


    long long cnt = 0, cnt_all = 0; double t = GetTime();

	DefineMapV(count) { cnt += v.count;};
	vertexMap(All, CTrueV, count);
    cnt_all = Sum(cnt);

    print( "number of 3-path=%lld\ntotal time=%0.3lf secs\n", cnt_all, t);
	return 0;
}
