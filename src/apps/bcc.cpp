#include "../core/api.h"
#define GT(A,B) (A.d>B.d || (A.d==B.d && A.cid>B.cid))

int main(int argc, char *argv[]) {
	VertexType(int,id, int,d,int,cid,int,p,int,dis);
	SetDataset(argv[1], argv[2]);
	SetSynAll(true);

	DefineMapV(init) {v.id = id(v); v.cid = id(v); v.d = deg(v); v.dis = -1; v.p = -1;};
	vertexSubset A = vertexMap(All, CTrueV, init);

	for(int len = Size(A), i = 0; len > 0; len = Size(A),++i) {
		print("CC Round %d: size=%d\n", i, len);

		DefineFE(check1) {return GT(s, d);};
		DefineMapE(update1) {d.cid = s.cid; d.d = s.d; return d;};
		DefineReduce(reduce1) {if (GT(s, d)) {d.cid = s.cid; d.d = s.d;} return d;};
		A = edgeMap(A, EU, check1, update1, CTrueV, reduce1);
	}

	vector<vertexSubset> v_bfs;
	DefineFV(filter1) {return v.cid == id(v);};
	DefineMapV(local1) {v.dis = 0; return v;};
	A = vertexMap(All, filter1, local1);
	for(int len = A.size(), i = 1; len > 0; len = A.size(), ++i) {
		print("BFS Round %d: size=%d\n", i, len);
		v_bfs.push_back(A);

		DefineMapE(update2) {d.dis = i; return d;};
		DefineFV(cond2) {return (v.dis == -1);};
		A = edgeMap(A, EU, CTrueE, update2, cond2, update2);
	}

	All.Pull(v.p=-1;for_nb(if(nb.dis==v.dis-1) {v.p=nb_id;break;}));
	DefineFE(check3) {return (s.dis == d.dis - 1);};
	DefineMapE(update3) {d.p = s.id; return d;};
	DefineFV(cond3) {return (v.p == -1);};
	DefineReduce(reduce3) {d = s; return d;};
	edgeMap(All, EU, check3, update3, cond3, reduce3);

	print( "Joining Edges...\n" );
	union_find f(n_vertex), cc;

	DefineMapV(join_edges) {
		for_nb(if(nb_id>v_id && v.p!=nb_id && nb.p!=v_id) {
			int a = nb_id, b = v_id;
			union_f(f,a,b);
			while(a != b) {
				int da=get(a).dis, db=get(b).dis, pa=get(a).p, pb=get(b).p;
				if(da>=db) {if(pa!=pb)union_f(f,pa,a); a=pa;}
				if(db>=da) {if(pa!=pb)union_f(f,pb,b); b=pb;}
			}
		})
	};
	vertexMap(All, CTrueV, join_edges, false);

	print( "Reducing...\n" );
	Reduce(f,cc,for_i(union_f(cc,f[i],i)));

	double t = GetTime();
	vector<int> cnt(n_vertex,0);
	int nc = 0, lc = 0;
	for(int i = 0; i < n_vertex; ++i) {
		int fi = get_f(cc,cc[i]);
		if(cnt[fi] == 1 ) ++nc; ++cnt[fi]; lc = max(lc, cnt[fi]);
	}

	print( "num_bcc=%d, max_bcc=%d\ntotal time=%0.3lf secs\n", nc, lc+1, t);

	return 0;
}
