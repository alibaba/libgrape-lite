#ifndef GFS_H_
#define GFS_H_

#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <vector>

#define BUFSIZE 4096
#define BUFFERED 1
#define NOBUFFER 0
#define MAX_MEM_EDG 100000000
#define MAX_NODE_ID 1000000000


#define NEG (-2147483648)
#define ALL 2147483647
#define ID2IN(V) ((V)|NEG)
#define IN2ID(V) ((V)&ALL)

typedef long long fileint;

using namespace std;


class MyReadFile {
private:
	int fin;
	string path;
	char buf[BUFSIZE];
	fileint pos;
	fileint buf_pos;
	int mode;//0:direct 1:buffered
	fileint total_io;
public:
	MyReadFile();
	MyReadFile(const string &path);
	MyReadFile(const string &path, int mode);
	~MyReadFile();
	void set_file(const string &path);
	bool fopen( int mode );
	bool fopen(const string &path, int mode);
	void fseek( fileint pos );
	void fread( void *dat, fileint size );
	void fclose();
	fileint get_total_io();
};

class MyWriteFile {
private:
	int fout;
	string path;
	char buf[BUFSIZE];
	fileint pos;
	fileint buf_pos;
	int mode;//0:direct 1:buffered
	fileint total_io;
public:
	MyWriteFile();
	MyWriteFile(const string &path);
	MyWriteFile(const string &path, int mode);
	~MyWriteFile();
	void set_file(const string &path);
	void fcreate( fileint size );
	void fcreate( fileint size, char ch );
	void fflush();
	bool fopen( int mode );
	bool fopen(const string &path, int mode);
	void fseek( fileint pos );
	//void fread( void *dat, fileint size );
	void fwrite( void *dat, fileint size );
	void fset( int pos_in_byte ); //set the pos'th bit in the current byte
	void fclear( int pos_in_byte );//clear the pos'th bit in the current byte
	void fclose();
	fileint get_total_io();
};

class Edge {
public:
	int a, b;
public:
	Edge();
	Edge(int a, int b);

public:
	bool operator < ( const Edge &e ) const;
};

class WEdge {
public:
	int a, b;
	float w;
public:
	WEdge();
	WEdge(int a, int b, float w);

public:
	bool operator < (const WEdge &e) const;
};

class GFS {
private:
	static void save_tmp_edge(string tmp_dir, Edge *mem_edge, int &m, int &n);
	static void merge(string tmp_dir, string output_dir, int n, int nx);
	static void save_buf_edge(FILE *fout, int u, vector<int> &adj, vector<fileint> &pos, fileint &nowpos);
	static void gen_bipartite_nx(string file, int num_cnt, int &nx, int &ny);
	static int sort_edge_tmp(string file, string tmp_dir, int &nx, bool directed, bool bipartite, long max_mem_edge);
	static void output_idx(vector<fileint> &pos, int nx, string output_dir);
	static int get_node_id(int *m_nid, int &n_node, int p);

private:
	static int sort_edge_tmp_w(string file, string tmp_dir, int &nx, bool directed, bool bipartite, long max_mem_edge);
	static void merge_w(string tmp_dir, string output_dir, int n, int nx);
	static void save_tmp_edge_w(string tmp_dir, WEdge *mem_edge, int &m, int &n);
	static void save_buf_edge_w(FILE *fout, FILE *fout_w, int u, vector<int> &adj, vector<float> &adj_w, vector<fileint> &pos, fileint &nowpos);

public:
	static int load_nbr(MyReadFile &rf, MyReadFile &rf_idx, int p, void *nbr, int &din, bool load_din);
	static int load_w(MyReadFile &rf_w, MyReadFile &rf_idx, int p, void *nbr_w, bool load_din);

	static int get_deg(MyReadFile &rf_idx, int p);
	static void get_graph_info(MyReadFile &rf, int &n, int &max_deg, int &nx);
	static void get_graph_info(const string &path, int &n, int &max_deg, int &nx, bool &weighted);
	static void txt2bin(string path,  string tmp_dir, string output_dir, bool directed = false, bool bipartite = false, long max_mem_edge = MAX_MEM_EDG );
	static void to_path(string &path);
	static void txt2bin_w(string path,  string tmp_dir, string output_dir, bool directed = false, bool bipartite = false, long max_mem_edge = MAX_MEM_EDG);

public:
	static bool get_edge(char *line, int &a, int &b, float &w, int num_cnt);
	static int get_num_cnt(string path);
};

//implementation

MyReadFile::MyReadFile( const string &path ) {
	this->path = path;
	fin = -1;
	buf_pos = -1;
	pos = 0;
	mode = BUFFERED;
	total_io = 0;
}

MyReadFile::MyReadFile() {
	fin = -1;
	buf_pos = -1;
	pos = 0;
	mode = BUFFERED;
	total_io = 0;
}

MyReadFile::MyReadFile(const string &path, int mode) {
	fopen( path, mode );
}

void MyReadFile::set_file( const string &path ) {
	this->path = path;
	fin = -1;
	buf_pos = -1;
	total_io = 0;
}

MyReadFile::~MyReadFile() {
	fclose();
}

bool MyReadFile::fopen(int mode) {
	this->mode = mode;
	fin = open( path.c_str(), O_RDONLY );
	if( fin < 0 )
		return false;
	pos = 0;
	if( mode == BUFFERED ) {
		buf_pos = 0;
		ssize_t unused __attribute__((unused)) = read( fin, buf, BUFSIZE );
		++total_io;
	}
	return true;
}


bool MyReadFile::fopen(const string &path, int mode) {
	set_file( path );
	return fopen( mode );
}

void MyReadFile::fclose() {
	if( fin >= 0 ) {
		close( fin );
		fin = -1;
		buf_pos = -1;
	}
}

void MyReadFile::fseek( fileint pos ) {
	if( mode == BUFFERED ) {
		if( pos >= buf_pos + BUFSIZE || pos < buf_pos ) {
			buf_pos = (pos / BUFSIZE) * BUFSIZE;
			lseek( fin, buf_pos, SEEK_SET );
			ssize_t unused __attribute__((unused)) = read( fin, buf, BUFSIZE );
			++total_io;
		}
	} else {
		lseek( fin, pos, SEEK_SET );
	}

	this->pos = pos;
}

void MyReadFile::fread( void *dat, fileint size ) {
	if( mode == BUFFERED ) {
		if( pos + size < buf_pos + BUFSIZE ) {
			memcpy( dat, buf + (pos - buf_pos), size );
		}
		else {
			memcpy( dat, buf + (pos-buf_pos), BUFSIZE-(pos-buf_pos) );
			while( buf_pos + BUFSIZE <= pos + size ) {
				buf_pos += BUFSIZE;
				ssize_t unused __attribute__((unused)) = read( fin, buf, BUFSIZE );
				++total_io;
				if( buf_pos + BUFSIZE > pos + size && buf_pos != pos + size ) {
					memcpy( (char*)dat + (buf_pos - pos), buf, pos + (size - buf_pos) );
				}
				else if( buf_pos != pos + size ) {
					memcpy( (char*)dat + (buf_pos - pos), buf, BUFSIZE );
				}
			}
		}
	} else {
		ssize_t unused __attribute__((unused)) = read( fin, dat, size );
		++total_io;
	}

	pos += size;
}

fileint MyReadFile::get_total_io() {
	return total_io;
}

//////// MyWriteFile
MyWriteFile::MyWriteFile( const string &path ) {
	this->path = path;
	fout = -1;
	buf_pos = -1;
	pos = 0;
	mode = BUFFERED;
	total_io = 0;
}

MyWriteFile::MyWriteFile() {
	fout = -1;
	buf_pos = -1;
	pos = 0;
	mode = BUFFERED;
	total_io = 0;
}

MyWriteFile::MyWriteFile(const string &path, int mode) {
	fopen( path, mode );
}

void MyWriteFile::set_file( const string &path ) {
	this->path = path;
	fout = -1;
	buf_pos = -1;
	total_io = 0;
}

MyWriteFile::~MyWriteFile() {
	fclose();
}

void MyWriteFile::fcreate( fileint size ) {
	size = ((size+BUFSIZE-1) / BUFSIZE + 1)* BUFSIZE;

	fout = open( path.c_str(), O_WRONLY | O_CREAT, S_IREAD|S_IWRITE );
	char* tmp = new char[BUFSIZE];
	memset( tmp, -1, BUFSIZE );
	for( fileint i = 0; i < size; i += BUFSIZE ) {
		ssize_t unused __attribute__((unused)) = write( fout, tmp, BUFSIZE );
		++total_io;
	}
	close( fout );
	fout = -1;
	delete[] tmp;
}

void MyWriteFile::fcreate( fileint size, char ch ) {
	size = ((size+BUFSIZE-1) / BUFSIZE + 1)* BUFSIZE;

	fout = open( path.c_str(), O_WRONLY | O_CREAT, S_IREAD|S_IWRITE );
	char* tmp = new char[BUFSIZE];
	memset( tmp, ch, BUFSIZE );
	for( fileint i = 0; i < size; i += BUFSIZE ) {
		ssize_t unused __attribute__((unused)) = write( fout, tmp, BUFSIZE );
		++total_io;
	}
	close( fout );
	fout = -1;
	delete[] tmp;
}

void MyWriteFile::fflush() {
	if( mode == BUFFERED ) {
		lseek( fout, buf_pos, SEEK_SET );
		ssize_t unused __attribute__((unused)) = write( fout, buf, BUFSIZE );
		++total_io;
	}
}

bool MyWriteFile::fopen( int mode ) {
	this->mode = mode;
	pos = 0;
	if( mode == BUFFERED ) {
		buf_pos = 0;
		fout = open( path.c_str(), O_RDWR );
		if( fout < 0 )
			return false;
		ssize_t unused __attribute__((unused)) = read( fout, buf, BUFSIZE );
		++total_io;

	} else {
		fout = open( path.c_str(), O_WRONLY );
		if( fout < 0 )
			return false;
	}
	return true;
}

bool MyWriteFile::fopen(const string &path, int mode) {
	set_file( path );
	return fopen( mode );
}

void MyWriteFile::fclose() {
	if( fout < 0 )
		return;
	if( mode == BUFFERED ) {
		lseek( fout, buf_pos, SEEK_SET );
		ssize_t unused __attribute__((unused)) = write( fout, buf, BUFSIZE );
		++total_io;
	}

	close( fout );
	fout = -1;
	buf_pos = -1;
}

void MyWriteFile::fseek( fileint pos ) {
	if( mode == BUFFERED ) {
		if( pos >= buf_pos + BUFSIZE || pos < buf_pos ) {
			lseek( fout, buf_pos, SEEK_SET );
			ssize_t unused __attribute__((unused)) = write( fout, buf, BUFSIZE );
			++total_io;

			buf_pos = (pos / BUFSIZE) * BUFSIZE;
			lseek( fout, buf_pos, SEEK_SET );
			unused = read( fout, buf, BUFSIZE );
			++total_io;
		}
	} else {
		lseek( fout, pos, SEEK_SET );
	}

	this->pos = pos;
}

void MyWriteFile::fwrite( void *dat, fileint size ) {
	if( mode == BUFFERED ) {
		if( pos + size < buf_pos + BUFSIZE ) {
			memcpy( buf + (pos-buf_pos),  dat,  size );
		} else {
			memcpy( buf + (pos-buf_pos), dat, BUFSIZE-(pos-buf_pos) );
			while( buf_pos + BUFSIZE <= pos + size ) {
				lseek( fout, buf_pos, SEEK_SET );
				ssize_t unused __attribute__((unused)) = write( fout, buf, BUFSIZE );
				++total_io;

				buf_pos += BUFSIZE;
				unused = read( fout, buf, BUFSIZE );
				++total_io;

				if( buf_pos + BUFSIZE > pos + size && buf_pos != pos + size ) {
					memcpy( buf, (char*)dat+buf_pos-pos, pos+(size-buf_pos) );
				}
				else if( buf_pos != pos + size ) {
					memcpy( buf, (char*)dat + (buf_pos - pos), BUFSIZE );
				}
			}
		}
	} else {
		ssize_t unused __attribute__((unused)) = write( fout, dat, size );
		++total_io;
	}
	pos += size;
}

void MyWriteFile::fset( int pos_in_byte ) {
	if( pos_in_byte < 0 || pos_in_byte >= 8 )
		return;
	if( mode == BUFFERED )
		buf[pos-buf_pos] |= (1<<pos_in_byte);
	else {
		char tmp;
		ssize_t unused __attribute__((unused)) = read( fout, &tmp, sizeof(char) );
		tmp |= (1<<pos_in_byte);
		unused = write( fout, &tmp, sizeof(char) );
		++total_io;
	}
}

void MyWriteFile::fclear( int pos_in_byte ) {
	if( pos_in_byte < 0 || pos_in_byte >= 8 )
		return;
	if( mode == BUFFERED )
		buf[pos-buf_pos] &= (~(1<<pos_in_byte));
	else {
		char tmp;
		ssize_t unused __attribute__((unused)) = read( fout, &tmp, sizeof(char) );
		tmp &= (~(1<<pos_in_byte));
		unused = write( fout, &tmp, sizeof(char) );
		++total_io;
	}
}

fileint MyWriteFile::get_total_io() {
	return total_io;
}

//////Edge
Edge::Edge(int a, int b) {
	this->a = a;
	this->b = b;
}

Edge::Edge(){
	this->a = 0;
	this->b = 0;
}

bool Edge::operator < (const Edge &e) const {
	if( a < e.a )
		return true;
	if( a > e.a )
		return false;
	return b < e.b;
}

//////WEdge
WEdge::WEdge(int a, int b, float w) {
	this->a = a;
	this->b = b;
	this->w = w;
}

WEdge::WEdge(){
	this->a = 0;
	this->b = 0;
	this->w = 0;
}

bool WEdge::operator < (const WEdge &e) const {
	if( a < e.a )
		return true;
	if( a > e.a )
		return false;
	if( b < e.b )
		return true;
	if( b > e.b )
		return false;
	return w < e.w;
}

//////GFS

void GFS::save_tmp_edge(string tmp_dir, Edge *mem_edge, int &m, int &n){
	printf( "sorting\n" );
	sort( mem_edge, mem_edge + m );
	char st[1024];
	sprintf( st, "%sedges_%d", tmp_dir.c_str(), n );
	printf( "creating file %s\n", st );

	FILE* fout = fopen( st, "wb" );
	fwrite( &m, sizeof(int), 1, fout );
	for( int i = 0; i < m; ++i )
		fwrite( mem_edge+i, sizeof(Edge), 1, fout );
	fclose( fout );

	++n;
	m = 0;
	printf( "finished write mem edge\n" );
}


void GFS::save_tmp_edge_w(string tmp_dir, WEdge *mem_edge, int &m, int &n){
	printf( "sorting\n" );
	sort( mem_edge, mem_edge + m );
	char st[1024];
	sprintf( st, "%sedges_%d", tmp_dir.c_str(), n );
	printf( "creating file %s\n", st );

	FILE* fout = fopen( st, "wb" );
	fwrite( &m, sizeof(int), 1, fout );
	for( int i = 0; i < m; ++i )
		fwrite( mem_edge+i, sizeof(WEdge), 1, fout );
	fclose( fout );

	++n;
	m = 0;
	printf( "finished write mem edge\n" );
}

void GFS::save_buf_edge_w(FILE *fout, FILE *fout_w, int u, vector<int> &adj, vector<float> &adj_w, vector<fileint> &pos, fileint &nowpos) {
	while( (int) pos.size() < u )
		pos.push_back( nowpos );
	pos.push_back( nowpos );

	int len = (int) adj.size();

	for( int i = 0; i < len; ++i ) {
		int val = adj[i];
		float w = adj_w[i];
		fwrite( &val, sizeof(int), 1, fout );
		fwrite( &w, sizeof(float), 1, fout_w );
		nowpos += sizeof(int);
	}

	adj.clear(); adj_w.clear();
}

void GFS::save_buf_edge(FILE *fout, int u, vector<int> &adj, vector<fileint> &pos, fileint &nowpos) {
	while( (int) pos.size() < u )
		pos.push_back( nowpos );
	pos.push_back( nowpos );

	int len = (int) adj.size();

	for( int i = 0; i < len; ++i ) {
		int val = adj[i];
		fwrite( &val, sizeof(int), 1, fout );
		nowpos += sizeof(int);
	}

	adj.clear();
}

void GFS::output_idx(vector<fileint> &pos, int nx, string output_dir){
	char st[1024];
	sprintf( st, "%sgraph.idx", output_dir.c_str() );

	FILE *fout = fopen(st, "wb");
	int len = (int)pos.size() - 1;
	fwrite( &len, sizeof(int), 1, fout);

	for( int i = 0; i <= len; ++i )  {
		fileint nowpos = pos[i];
		fwrite( &nowpos, sizeof(fileint), 1, fout );
	}

	fwrite( &nx, sizeof(int), 1, fout );
	fclose( fout );
}

void GFS::merge_w(string tmp_dir, string output_dir, int n, int nx){
	long long n_edge = 0;
	printf( "merging edges, n_file=%d ...\n", n );

	FILE** fin = new FILE*[n];
	char st[1024];
	for( int i = 0; i < n; ++i ) {
		sprintf( st, "%sedges_%d", tmp_dir.c_str(), i );
		fin[i] = fopen( st, "rb" );
	}

	vector<fileint> pos;
	vector<int> adj;
	vector<float> adj_w;
	fileint nowpos = 0;

	sprintf( st, "%sgraph.dat", output_dir.c_str() );
	FILE *fout = fopen(st, "wb");

	sprintf( st, "%sgraph.w", output_dir.c_str() );
	FILE *fout_w = fopen(st, "wb");

	WEdge *e = new WEdge[n];
	int *len = new int[n];
	int *nowp = new int[n];

	for( int i = 0; i < n; ++i ) {
		size_t unused __attribute__((unused)) = fread( len+i, sizeof(int), 1, fin[i] );
		unused = fread( e+i, sizeof(WEdge), 1, fin[i] );
		nowp[i] = 1;
	}

	fileint cnt = 0;
	WEdge pre(-1,-1,0);
	while( true ){
		WEdge now(-1,-1,0);
		int p = -1;
		for( int i = 0; i < n; ++i)
			if( e[i].a >= 0 )
				if( now.a < 0 || e[i] < now ) {
					p = i;
					now = e[i];
				}
		if( now.a < 0 ) break;
		if( nowp[p] == len[p] ) e[p] = WEdge(-1,-1,0);
		else {
			size_t unused __attribute__((unused)) = fread(e+p, sizeof(WEdge), 1, fin[p]);
			++nowp[p];
		}

		if( now.a != pre.a && pre.a >= 0 ) {
			n_edge += adj.size();
			save_buf_edge_w( fout, fout_w, pre.a, adj, adj_w, pos, nowpos );
		}

		//this condition can be changed depending on requirements
		if( (adj.size() == 0 || now.b != pre.b) && (now.a != (now.b & ALL)) ) {
			adj.push_back( now.b );
			adj_w.push_back( now.w );
		}
		pre = now;

		if( ++cnt % 20000000 == 0 ) printf( "[%lld]\n", cnt );
	}

	if( adj.size() > 0 ) {
		n_edge += adj.size();
		save_buf_edge_w( fout, fout_w, pre.a, adj, adj_w, pos, nowpos );
	}

	++pre.a;
	save_buf_edge_w( fout, fout_w, pre.a, adj,adj_w,  pos, nowpos );

	delete[] e;
	delete[] len;
	delete[] nowp;
	fclose( fout );
	fclose( fout_w );

	for( int i = 0; i < n; ++i )
		fclose( fin[i] );

	delete[] fin;
	printf( "\nm=%lld\n", n_edge );
	output_idx( pos, nx, output_dir );
}

void GFS::merge(string tmp_dir, string output_dir, int n, int nx){
	long long n_edge = 0;
	printf( "merging edges...\n" );

	FILE** fin = new FILE*[n];
	char st[1024];
	for( int i = 0; i < n; ++i ) {
		sprintf( st, "%sedges_%d", tmp_dir.c_str(), i );
		fin[i] = fopen( st, "rb" );
	}

	vector<fileint> pos;
	vector<int> adj;
	fileint nowpos = 0;

	sprintf( st, "%sgraph.dat", output_dir.c_str() );

	FILE *fout = fopen(st, "wb");
	Edge *e = new Edge[n];
	int *len = new int[n];
	int *nowp = new int[n];

	for( int i = 0; i < n; ++i ) {
		size_t unused __attribute__((unused)) = fread( len+i, sizeof(int), 1, fin[i] );
		unused = fread( e+i, sizeof(Edge), 1, fin[i] );
		nowp[i] = 1;
	}

	fileint cnt = 0;
	Edge pre(-1,-1);
	while( true ){
		Edge now(-1,-1);
		int p = -1;
		for( int i = 0; i < n; ++i)
			if( e[i].a >= 0 )
				if( now.a < 0 || e[i] < now ) {
					p = i;
					now = e[i];
				}
		if( now.a < 0 ) break;
		if( nowp[p] == len[p] ) e[p] = Edge(-1,-1);
		else {
			size_t unused __attribute__((unused)) = fread(e+p, sizeof(Edge), 1, fin[p]);
			++nowp[p];
		}

		if( now.a != pre.a && pre.a >= 0 ) {
			n_edge += adj.size();
			save_buf_edge( fout, pre.a, adj, pos, nowpos );
		}

		//this condition can be changed depending on requirements
		if( (adj.size() == 0 || now.b != pre.b) && (now.a != (now.b & ALL)) )
			adj.push_back( now.b );
		pre = now;

		if( ++cnt % 20000000 == 0 ) printf( "[%lld]\n", cnt );
	}

	if( adj.size() > 0 ) {
		n_edge += adj.size();
		save_buf_edge( fout, pre.a, adj, pos, nowpos );
	}

	++pre.a;
	save_buf_edge( fout, pre.a, adj, pos, nowpos );

	delete[] e;
	delete[] len;
	delete[] nowp;
	fclose( fout );

	for( int i = 0; i < n; ++i )
		fclose( fin[i] );

	delete[] fin;
	printf( "\nm=%lld\n", n_edge );
	output_idx( pos, nx, output_dir );
}

int GFS::get_node_id(int *m_nid, int &n_node, int p){
	if( m_nid[p] < 0 ) {
		m_nid[p] = n_node++;
		return n_node-1;
	}
	return m_nid[p];
}

void GFS::gen_bipartite_nx(string file, int num_cnt, int &nx, int &ny) {
	printf( "Scanning for bipartite graph...\n" );
	FILE *fin = fopen(file.c_str(), "r");
	char line[1024]; int cnt = 0, cnt2=0;
	nx = 0; ny = 0;
	while(fgets(line, 1024, fin)) {
		int u, v; float w;
		if( !get_edge(line, u, v, w, num_cnt) ) continue;

		++cnt;
		if( cnt % 1000000 == 0 ) {
			cnt = 0; ++cnt2;
			printf( "[%dM]", cnt2 );
		}

		nx = max(nx, u+1);
		ny = max(ny, v+1);
	}
	printf( "\nnx=%d,ny=%d\n", nx, ny );
	fclose(fin);
}

int GFS::sort_edge_tmp(string file, string tmp_dir, int &nx, bool directed, bool bipartite, long max_mem_edge) {
	Edge *mem_edge = new Edge[max_mem_edge];

	nx = 0;
	int n = 0, m = 0, num_cnt = get_num_cnt(file), ny=0;
	char line[1024];
	int cnt = 0, cnt2 = 0, my_nx=0;

	if(bipartite) {
		if(!directed) {
			gen_bipartite_nx(file, num_cnt, nx, ny);
			my_nx = nx;
		}
		else {
			FILE *fin = fopen(file.c_str(), "r" );
			int unused __attribute__((unused)) = fscanf(fin, "%d", &my_nx);
			fclose(fin);
		}
	}

	printf( "Processing %s, num_cnt=%d\n", file.c_str(), num_cnt );
	FILE *fin = fopen( file.c_str(), "r" );

	cnt = 0, cnt2 = 0;
	while( fgets(line, 1024, fin) ) {
		if( line[0] < '0' || line[0] > '9' ) continue;

		int u, v; float w;
		if( !get_edge(line, u, v, w, num_cnt) ) continue;
		if( u < 0 || v < 0 ) continue;

		++cnt;
		if( cnt % 1000000 == 0 ) {
			cnt = 0; ++cnt2;
			printf( "[%dM]", cnt2 );
		}

		mem_edge[m].a = u;
		mem_edge[m].b = v + nx;
		++m;
		mem_edge[m].a = v + nx;
		mem_edge[m].b = directed ? ID2IN(u):u;
		++m;

		if( m >= max_mem_edge ) save_tmp_edge( tmp_dir, mem_edge, m, n );
	}

	fclose( fin );

	if( m > 0 ) save_tmp_edge( tmp_dir, mem_edge, m, n );
	printf( "finish sorting edge tmp\n" );
	delete[] mem_edge;
	nx = my_nx;

	return n;
}


int GFS::sort_edge_tmp_w(string file, string tmp_dir, int &nx, bool directed, bool bipartite, long max_mem_edge) {
	WEdge *mem_edge = new WEdge[max_mem_edge+1];

	nx = 0;
	int n = 0, m = 0, num_cnt = get_num_cnt(file), ny=0;
	char line[1024];
	int cnt = 0, cnt2 = 0, my_nx = 0;
	if(num_cnt < 3) {
		printf( "Format error, num_cnt=%d\n", num_cnt );
		return -1;
	}

	if(bipartite) {
		if(!directed) {
			gen_bipartite_nx(file, num_cnt, nx, ny);
			my_nx = nx;
		}
		else {
			FILE *fin = fopen(file.c_str(), "r" );
			int unused __attribute__((unused)) = fscanf(fin, "%d", &my_nx);
			fclose(fin);
		}
	}

	printf( "Processing %s, num_cnt=%d\n", file.c_str(), num_cnt );
	FILE *fin = fopen( file.c_str(), "r" );

	cnt = 0, cnt2 = 0;
	while( fgets(line, 1024, fin) ) {
		if( line[0] < '0' || line[0] > '9' ) continue;

		int u, v; float w;
		if( !get_edge(line, u, v, w, num_cnt) ) continue;
		if( u < 0 || v < 0 ) continue;

		++cnt;
		if( cnt % 1000000 == 0 ) {
			cnt = 0; ++cnt2;
			printf( "[%dM]", cnt2 );
		}

		mem_edge[m].a = u;
		mem_edge[m].b = v + nx;
		mem_edge[m].w = w;
		++m;
		mem_edge[m].a = v + nx;
		mem_edge[m].b = directed ? ID2IN(u):u;
		mem_edge[m].w = w;
		++m;

		if( m >= max_mem_edge ) save_tmp_edge_w( tmp_dir, mem_edge, m, n );
	}

	fclose( fin );

	if( m > 0 ) save_tmp_edge_w( tmp_dir, mem_edge, m, n );
	printf( "finish sorting edge tmp\n" );
	delete[] mem_edge;

	nx = my_nx;
	return n;
}

int GFS::get_deg(MyReadFile &rf_idx, int p) {
	fileint pos = sizeof(int) + p * sizeof(fileint);
	rf_idx.fseek(pos);
	fileint nowpos, nxtpos;

	rf_idx.fread( &nowpos, sizeof(fileint) );
	rf_idx.fread( &nxtpos, sizeof(fileint) );

	return (int) (nxtpos-nowpos) / sizeof(int);
}

int GFS::load_nbr(MyReadFile &rf, MyReadFile &rf_idx, int p, void *nbr, int &din, bool load_din) {
	fileint pos = load_din? sizeof(int) + p * (sizeof(fileint)+sizeof(int)) : sizeof(int) + p * sizeof(fileint);
	rf_idx.fseek(pos);
	fileint nowpos, nxtpos;

	rf_idx.fread( &nowpos, sizeof(fileint) );
	if(load_din) rf_idx.fread( &din, sizeof(int) );
	rf_idx.fread( &nxtpos, sizeof(fileint) );

	int n_nbr = (int) ((nxtpos-nowpos) / sizeof(int));

	rf.fseek(nowpos);
	rf.fread( nbr, sizeof(int) * n_nbr );

	return n_nbr;
}


int GFS::load_w(MyReadFile &rf_w, MyReadFile &rf_idx, int p, void *nbr_w, bool load_din) {
	fileint pos = load_din? sizeof(int) + p * (sizeof(fileint)+sizeof(int)) : sizeof(int) + p * sizeof(fileint);
	rf_idx.fseek(pos);
	fileint nowpos, nxtpos;

	rf_idx.fread( &nowpos, sizeof(fileint) );
	if(load_din) {int din; rf_idx.fread( &din, sizeof(int) );}
	rf_idx.fread( &nxtpos, sizeof(fileint) );

	int n_nbr = (int) ((nxtpos-nowpos) / sizeof(float));

	rf_w.fseek(nowpos);
	rf_w.fread( nbr_w, sizeof(float) * n_nbr );

	return n_nbr;
}

void GFS::get_graph_info(MyReadFile &rf, int &n, int &max_deg, int &nx) {
	rf.fread( &n, sizeof(int) );
	fileint nowpos, nxtpos;

	max_deg = 0;
	rf.fread( &nowpos, sizeof(fileint) );
	for( int i = 0; i < n; i ++ ) {
		rf.fread( &nxtpos, sizeof(fileint) );
		int len = (int) (nxtpos - nowpos) / sizeof(int);
		max_deg = len > max_deg ? len :max_deg;
		nowpos =nxtpos;
	}
	rf.fread( &nx, sizeof(int) );
}

void GFS::get_graph_info(const string &path, int &n, int &max_deg, int &nx, bool &weighted) {
	string file_idx = path + "graph.idx";

	MyReadFile rf( file_idx );
	rf.fopen( BUFFERED );
	get_graph_info(rf, n, max_deg, nx);
	rf.fclose();

	weighted = false;
	FILE *fin = fopen((path+"graph.w").c_str(), "rb");
	if(fin) {
		weighted = true;
		fclose(fin);
	}
}

void GFS::to_path(string &path) {
	if(path.c_str()[path.size()-1] != '/') path=path+"/";
}

void GFS::txt2bin(string path, string tmp_dir, string output_dir, bool directed, bool bipartite, long max_mem_edge) {
	to_path(path); to_path(tmp_dir); to_path(output_dir);
	string file = path + "graph.txt";
	int nx = 0;
	int n = sort_edge_tmp(file, tmp_dir, nx, directed, bipartite, max_mem_edge);
	merge( tmp_dir, output_dir, n, nx );
}


void GFS::txt2bin_w(string path, string tmp_dir, string output_dir, bool directed, bool bipartite, long max_mem_edge) {
	to_path(path); to_path(tmp_dir); to_path(output_dir);
	string file = path + "graph.txt";
	int nx = 0;
	int n = sort_edge_tmp_w(file, tmp_dir, nx, directed, bipartite, max_mem_edge);
	if(n>=0) merge_w( tmp_dir, output_dir, n, nx );
}



bool GFS::get_edge(char *line, int &a, int &b, float &w, int num_cnt) {
	if( !isdigit(line[0]) ) return false;
	vector<char*> v_num;
	int len = (int) strlen(line);
	for( int i = 0; i < len; ++i )
		if( !isdigit(line[i]) && line[i] != '.' && line[i] != '-') line[i] = '\0';
		else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
	if( (int) v_num.size() != num_cnt ) return false;
	sscanf( v_num[0], "%d", &a );
	sscanf( v_num[1], "%d", &b );
	w=0;
	if(num_cnt >= 3) sscanf(v_num[2], "%f", &w);
	return true;
}

int GFS::get_num_cnt(string file) {
	FILE *fin = fopen(file.c_str(), "r");
	char line[1024];
	int cnt = 0, min_cnt = 100;

	while( fgets( line, 1024, fin ) && cnt < 10 ) {
		if( !isdigit(line[0]) ) continue;
		vector<char*> v_num;
		int len = (int) strlen(line);
		for( int i = 0; i < len; ++i )
			if( !isdigit(line[i]) && line[i] != '.' ) line[i] = '\0';
			else if(i == 0 || !line[i-1]) v_num.push_back(line+i);
		if( (int) v_num.size() < 2 ) continue;
		min_cnt = min(min_cnt, (int) v_num.size());
		++cnt;
	}
	fclose( fin );
	return min_cnt;
}

#endif /* BIGGRAPH_SRC_GFS_H_ */
