#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(vector<int>,from, vector<int>,walk,vector<int>,from2,NONE);
	SetDataset(argv[1], argv[2]);

	DefineFV(filter) {return deg(v) > 0;};
	vertexSubset A = vertexMap(All, filter);

	DefineMapV(local) {v.walk.push_back(get_nb_id(rand()%deg(v)));};
	A = vertexMap(A, CTrueV, local);

	DefineMapV(init) {
		v.from.clear(); 
		v.from2.clear(); 
		return v;
	};

	DefineOutEdges(edges1) {
		vector<int> res; 
		res.clear(); 
		res.push_back(v.walk.back()); 
		return res;
	};
	DefineMapE(update1) {d.from.push_back(id(s));};
	DefineReduce(merge1) {reduce(d.from.clear(), insert(d.from, s.from))};

	DefineOutEdges(edges2) {return v.from;};
	DefineMapE(update2) {d.from2.push_back(get_nb_id(rand()%deg(s)));};
	DefineReduce(merge2) {d.walk.push_back(s.from2.front());};

	for(int i = 1; i < 5; ++i) {
		print( "Walk Step=%d\n", i );
		vertexSubset B = vertexMap(A, CTrueV, init);
		B = edgeMapSparse(A, edges1, CTrueE, update1, CTrueV, merge1);
		edgeMapSparse(B, edges2, CTrueE, update2, CTrueV, merge2);
	}

	//All.Gather(if (v_id > 10) return; printf( "%d:", v_id); for(auto &u:v.walk) printf( " %d", u); printf( "\n") );
	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
