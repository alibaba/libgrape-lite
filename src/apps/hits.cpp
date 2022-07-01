#include "../core/api.h"

int main(int argc, char *argv[]) {
	VertexType(float,auth,float,hub, float, auth1, float, hub1, ONE+TWO);
	SetDataset(argv[1], argv[2]);

	DefineMapV(init) {v.auth=1;v.hub=1;};
	vertexMap(All, CTrueV, init);

	double sa = 0, sh = 0, sa_all = 0, sh_all = 0;
	DefineMapV(local1) {v.auth1 = 0; v.hub1 = 0;};
	DefineMapE(update1) {d.auth1 += s.hub;};
	DefineMapE(update2) {d.hub1 += s.auth;};
	DefineMapV(local2) {v.auth = v.auth1; v.auth1 = 0; v.hub = v.hub1; v.hub1 = 0; sa += v.auth*v.auth; sh+= v.hub*v.hub;};
	DefineMapV(local3) {v.auth/=sa_all; v.hub/=sh_all;};

	for(int i = 0; i < 10; ++i) {
		print("Round %d\n", i);
		sa = 0, sh = 0, sa_all = 0, sh_all = 0;

		All.Pull(v.auth=0;v.hub=0;for_in(v.auth+=nb.hub);for_out(v.hub+=nb.auth); sa+=v.auth*v.auth; sh+=v.hub*v.hub, NONE);

		vertexMap(All, CTrueV, local1);
		edgeMapDense(All, ED, CTrueE, update1, CTrueV);
		edgeMapDense(All, ER, CTrueE, update2, CTrueV);

		vertexMap(All, CTrueV, local2);
		sa_all=sqrt(Sum(sa)); sh_all=sqrt(Sum(sh));

		vertexMap(All, CTrueV, local2);

	}

	//All.Gather(printf( "v=%d,auth=%0.5f,hub=%0.5f\n", v_id, v.auth, v.hub));
	print( "total time=%0.3lf secs\n", GetTime());
	return 0;
}
