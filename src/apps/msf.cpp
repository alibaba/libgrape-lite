#include "../core/api.h"
using E=pair<float,pair<int,int>>;

void kruskal(vector<E> &e, E *mst, int n) {
	union_find f(n);
	memset(mst,0,sizeof(E)*(n-1));
	sort(e.begin(),e.end());
	for(int i=0, p=0;i<(int)e.size() && p<n-1;++i) {
		int a=get_f(f,e[i].second.first), b=get_f(f,e[i].second.second);
		if(a != b) {union_f(f,a,b); mst[p++]=e[i];}
	}
}	

int main(int argc, char *argv[]) {
	VertexType();
	SetDataset(argv[1], argv[2]);

	vector<E> e, mst0(n_vertex-1), mst;
	DefineMapV(init) {
		for_nb(e.push_back(make_pair(weight,make_pair(nb_id, v_id))));
	};
	vertexMap(All, CTrueV, init);

	kruskal(e,mst0.data(),n_vertex);
	Reduce(mst0,mst,vector<E> e; e.assign(mst0,mst0+len); e.insert(e.end(),mst,mst+len); kruskal(e,mst,len+1));
	print( "total time=%0.3lf secs\n", GetTime());

	float wt = 0; for(auto &e:mst) wt+=e.first;
	print( "mst weight=%0.3f\n", wt );
	return 0;
}
