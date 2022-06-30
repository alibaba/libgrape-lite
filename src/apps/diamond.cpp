#include "../core/api.h"
#include <unordered_set>

int main(int argc, char *argv[]) {
	using Pair = pair<int,int>;
	VertexType(int,deg, int,id, vector<Pair>,out, int,count, ONE+TWO+THREE);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {
		v.count = 0; 
		v.id = id(v);
		v.deg = deg(v);
		return v;
	};
	vertexMap(All, CTrueV, init);

	//DefineFE(check) {return (s.deg > d.deg || (s.deg == d.deg && s.id > d.id));};
	DefineMapE(update) {d.out.push_back(make_pair(s.id, s.deg)); return d;};

	print( "Loading...\n" );
	edgeMapDense(All, EU, CTrueE, update, CTrueV);

	DefineMapV(count) {
		vector<int> cnt(n_vertex, 0);
		unordered_set<int> nghs;
		nghs.clear();
		for_nb(nghs.insert(nb_id));
		for_nb(
			for(auto &o:nb.out)
				if(o.second>v_deg||(o.second==v_deg && o.first>v_id)) 
				if (nghs.find(o.first) != nghs.end()) {
					int u = o.first;
					v.count += cnt[u]++;
				}
		);
	};

    print( "Computing...\n" );
	vertexMap(All, CTrueV, count, false);

    long long tt = 0, tt_all = 0; double t = GetTime();
	DefineMapV(agg) {tt += v.count;};
	vertexMap(All, CTrueV, agg);
    tt_all = Sum(tt);

    print( "number of diamonds=%lld\ntotal time=%0.3lf secs\n", tt_all, t);
	return 0;
}
