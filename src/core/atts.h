#ifndef ATTS_H_
#define ATTS_H_

#include <mpi.h>

#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <climits>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <typeinfo>

#include "buffer.h"

//declare
#define READ void read(BufManager &bm, int atts = -1)
#define WRITE void write(BufManager &bm, int atts = -1)
#define EQ(T) bool eq(const T& v, int atts = -1)
#define CP(T) void cp_from(T& v, int atts = -1)
#define SIZE int size(int atts = -1)
#define CMP(T) int cmp(const T& v, int atts = -1)
#define INIT void init()
#define CMP0 int cmp0(int atts = -1)

//define
#define READ_CP(TYPE) \
	void read_att(TYPE &att, BufManager &bm) {memcpy(&att, bm.buf+bm.pos, sizeof(TYPE)); bm.pos += sizeof(TYPE);}
#define WRITE_CP(TYPE) \
	void write_att(TYPE &att, BufManager &bm) {memcpy(bm.buf+bm.pos, &att, sizeof(TYPE)); bm.pos += sizeof(TYPE); bm.update();}
#define EQU_EXACT(TYPE) bool equ(const TYPE &a, const TYPE &b) {return a == b;}
#define EQU_INEXACT(TYPE) bool equ(const TYPE &a, const TYPE &b) {return fabs(a-b)<EPS;}
#define CP_ALL(TYPE) void cp_att(TYPE &a, const TYPE &b) {a=b;}
#define INIT_ATT(TYPE) void init_att(TYPE &att) {att=0;}
#define CMP0_ATT(TYPE) bool cmp0_att(TYPE &att) {return att==0;}
#define CMP0_ATT_INEXACT(TYPE) bool cmp0_att(TYPE &att) {return att<EPS && att>-EPS;}
#define GET_SIZE(TYPE) int get_size(TYPE &att) {return (int)sizeof(TYPE);}

//use
#define READATT(NUM,ATT) if(atts&NUM) read_att(ATT, bm);
#define WRITEATT(NUM,ATT) if(atts&NUM) write_att(ATT, bm);
#define EQATT(NUM,ATT) if(atts&NUM) if(!equ(ATT,v.ATT)) return false;
#define CPATT(NUM,ATT) if(atts&NUM) cp_att(ATT,v.ATT);
#define INITATT(ATT) init_att(ATT);
#define SZATT(NUM,ATT) ((atts&NUM)?get_size(ATT):0)
#define CMPATT(NUM,ATT) ((atts&NUM) && !equ(ATT,v.ATT) ? NUM : 0)
#define CMP0ATT(NUM,ATT) ((atts&NUM) && !cmp0_att(ATT) ? NUM : 0)

//define class
#define TUPLE0(NAME) \
	class NAME {\
	public:\
		NAME(){init();}\
		INIT {}\
		READ {}\
		WRITE {}\
		EQ(NAME) {return true;}\
		CP(NAME) {} \
		SIZE {return 0;} \
		CMP(NAME) {return 0;} \
		CMP0 {return 0;}\
	}

#define TUPLE1(NAME,T1,A1) \
	class NAME {\
	public:\
		T1 A1;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1)}\
		READ {READATT(ONE,A1)}\
		WRITE {WRITEATT(ONE,A1)}\
		EQ(NAME) {EQATT(ONE,A1) return true;}\
		CP(NAME) {CPATT(ONE,A1)} \
		SIZE {return SZATT(ONE,A1);} \
		CMP(NAME) {return CMPATT(ONE,A1);} \
		CMP0 {return CMP0ATT(ONE,A1);}\
	}

#define TUPLE2(NAME,T1,A1,T2,A2) \
	class NAME {\
	public:\
		T1 A1; T2 A2;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2)}\
		READ {READATT(ONE,A1) READATT(TWO,A2)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2)} \
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2);} \
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2);}\
	}

#define TUPLE3(NAME,T1,A1,T2,A2,T3,A3) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3)} \
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3);} \
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3);}\
	}

#define TUPLE4(NAME,T1,A1,T2,A2,T3,A3,T4,A4) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3; T4 A4;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3) INITATT(A4)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3) READATT(FOUR,A4)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3) WRITEATT(FOUR,A4)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) EQATT(FOUR,A4) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3) CPATT(FOUR,A4)}\
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3)+SZATT(FOUR,A4);}\
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3)+CMPATT(FOUR,A4);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3)+CMP0ATT(FOUR,A4);}\
	}

#define TUPLE5(NAME,T1,A1,T2,A2,T3,A3,T4,A4,T5,A5) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3; T4 A4; T5 A5;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3) INITATT(A4) INITATT(A5)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3) READATT(FOUR,A4) READATT(FIVE,A5)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3) WRITEATT(FOUR,A4) WRITEATT(FIVE,A5)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) EQATT(FOUR,A4) EQATT(FIVE,A5) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3) CPATT(FOUR,A4) CPATT(FIVE,A5)}\
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3)+SZATT(FOUR,A4)+SZATT(FIVE,A5);}\
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3)+CMPATT(FOUR,A4)+CMPATT(FIVE,A5);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3)+CMP0ATT(FOUR,A4)+CMP0ATT(FIVE,A5);}\
	}

#define TUPLE6(NAME,T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3; T4 A4; T5 A5; T6 A6;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3) INITATT(A4) INITATT(A5) INITATT(A6)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3) READATT(FOUR,A4) READATT(FIVE,A5) READATT(SIX,A6)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3) WRITEATT(FOUR,A4) WRITEATT(FIVE,A5) WRITEATT(SIX,A6)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) EQATT(FOUR,A4) EQATT(FIVE,A5) EQATT(SIX,A6) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3) CPATT(FOUR,A4) CPATT(FIVE,A5) CPATT(SIX,A6)}\
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3)+SZATT(FOUR,A4)+SZATT(FIVE,A5)+SZATT(SIX,A6);}\
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3)+CMPATT(FOUR,A4)+CMPATT(FIVE,A5)+CMPATT(SIX,A6);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3)+CMP0ATT(FOUR,A4)+CMP0ATT(FIVE,A5)+CMP0ATT(SIX,A6);}\
	}

#define TUPLE7(NAME,T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3; T4 A4; T5 A5; T6 A6; T7 A7;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3) INITATT(A4) INITATT(A5) INITATT(A6) INITATT(A7)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3) READATT(FOUR,A4) READATT(FIVE,A5) READATT(SIX,A6) READATT(SEVEN,A7)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3) WRITEATT(FOUR,A4) WRITEATT(FIVE,A5) WRITEATT(SIX,A6) WRITEATT(SEVEN,A7)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) EQATT(FOUR,A4) EQATT(FIVE,A5) EQATT(SIX,A6) EQATT(SEVEN,A7) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3) CPATT(FOUR,A4) CPATT(FIVE,A5) CPATT(SIX,A6) CPATT(SEVEN,A7)}\
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3)+SZATT(FOUR,A4)+SZATT(FIVE,A5)+SZATT(SIX,A6)+SZATT(SEVEN,A7);}\
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3)+CMPATT(FOUR,A4)+CMPATT(FIVE,A5)+CMPATT(SIX,A6)+CMPATT(SEVEN,A7);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3)+CMP0ATT(FOUR,A4)+CMP0ATT(FIVE,A5)+CMP0ATT(SIX,A6)+CMP0ATT(SEVEN,A7);}\
	}

#define TUPLE8(NAME,T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7,T8,A8) \
	class NAME {\
	public:\
		T1 A1; T2 A2; T3 A3; T4 A4; T5 A5; T6 A6; T7 A7; T8 A8;\
	public:\
		NAME(){init();}\
		INIT {INITATT(A1) INITATT(A2) INITATT(A3) INITATT(A4) INITATT(A5) INITATT(A6) INITATT(A7) INITATT(A8)}\
		READ {READATT(ONE,A1) READATT(TWO,A2) READATT(THREE,A3) READATT(FOUR,A4) READATT(FIVE,A5) READATT(SIX,A6) READATT(SEVEN,A7) READATT(EIGHT,A8)}\
		WRITE {WRITEATT(ONE,A1) WRITEATT(TWO,A2) WRITEATT(THREE,A3) WRITEATT(FOUR,A4) WRITEATT(FIVE,A5) WRITEATT(SIX,A6) WRITEATT(SEVEN,A7) WRITEATT(EIGHT,A8)}\
		EQ(NAME) {EQATT(ONE,A1) EQATT(TWO,A2) EQATT(THREE,A3) EQATT(FOUR,A4) EQATT(FIVE,A5) EQATT(SIX,A6) EQATT(SEVEN,A7) EQATT(EIGHT,A8) return true;}\
		CP(NAME) {CPATT(ONE,A1) CPATT(TWO,A2) CPATT(THREE,A3) CPATT(FOUR,A4) CPATT(FIVE,A5) CPATT(SIX,A6) CPATT(SEVEN,A7) CPATT(EIGHT,A8)}\
		SIZE {return SZATT(ONE,A1)+SZATT(TWO,A2)+SZATT(THREE,A3)+SZATT(FOUR,A4)+SZATT(FIVE,A5)+SZATT(SIX,A6)+SZATT(SEVEN,A7)+SZATT(EIGHT,A8);}\
		CMP(NAME) {return CMPATT(ONE,A1)+CMPATT(TWO,A2)+CMPATT(THREE,A3)+CMPATT(FOUR,A4)+CMPATT(FIVE,A5)+CMPATT(SIX,A6)+CMPATT(SEVEN,A7)+CMPATT(EIGHT,A8);} \
		CMP0 {return CMP0ATT(ONE,A1)+CMP0ATT(TWO,A2)+CMP0ATT(THREE,A3)+CMP0ATT(FOUR,A4)+CMP0ATT(FIVE,A5)+CMP0ATT(SIX,A6)+CMP0ATT(SEVEN,A7)+CMP0ATT(EIGHT,A8);}\
	}

#define VERTEX_T0() TUPLE0(VTYPE)
#define VERTEX_T1(...) TUPLE1(VTYPE,__VA_ARGS__)
#define VERTEX_T2(...) TUPLE2(VTYPE,__VA_ARGS__)
#define VERTEX_T3(...) TUPLE3(VTYPE,__VA_ARGS__)
#define VERTEX_T4(...) TUPLE4(VTYPE,__VA_ARGS__)
#define VERTEX_T5(...) TUPLE5(VTYPE,__VA_ARGS__)
#define VERTEX_T6(...) TUPLE6(VTYPE,__VA_ARGS__)
#define VERTEX_T7(...) TUPLE7(VTYPE,__VA_ARGS__)
#define VERTEX_T8(...) TUPLE8(VTYPE,__VA_ARGS__)

#define VERTEX_T1C(T1,A1,C) VERTEX_T1(T1,A1); GRAPH::critical_atts=C;
#define VERTEX_T2C(T1,A1,T2,A2,C) VERTEX_T2(T1,A1,T2,A2); GRAPH::critical_atts=C;
#define VERTEX_T3C(T1,A1,T2,A2,T3,A3,C) VERTEX_T3(T1,A1,T2,A2,T3,A3); GRAPH::critical_atts=C;
#define VERTEX_T4C(T1,A1,T2,A2,T3,A3,T4,A4,C) VERTEX_T4(T1,A1,T2,A2,T3,A3,T4,A4); GRAPH::critical_atts=C;
#define VERTEX_T5C(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,C) VERTEX_T5(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5); GRAPH::critical_atts=C;
#define VERTEX_T6C(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,C) VERTEX_T6(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6); GRAPH::critical_atts=C;
#define VERTEX_T7C(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7,C) VERTEX_T7(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7); GRAPH::critical_atts=C;
#define VERTEX_T8C(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7,T8,A8,C) VERTEX_T8(T1,A1,T2,A2,T3,A3,T4,A4,T5,A5,T6,A6,T7,A7,T8,A8); GRAPH::critical_atts=C;

#define GetVertexType(_0c, _1, _1c, _2, _2c, _3, _3c, _4, _4c, _5, _5c, _6, _6c, _7, _7c, _8, _8c, NAME, ...) NAME
#define VertexType(...) GetVertexType(__VA_ARGS__, VERTEX_T8C, VERTEX_T8, VERTEX_T7C, VERTEX_T7, VERTEX_T6C, VERTEX_T6, VERTEX_T5C, VERTEX_T5, \
	VERTEX_T4C, VERTEX_T4, VERTEX_T3C, VERTEX_T3, VERTEX_T2C, VERTEX_T2, VERTEX_T1C, VERTEX_T1, VERTEX_T0, ...)(__VA_ARGS__)

DEF_INT_TYPE(EQU_EXACT)
DEF_REAL_TYPE(EQU_INEXACT)
DEF_INT_TYPE(CMP0_ATT)
DEF_REAL_TYPE(CMP0_ATT_INEXACT)

DEF_ALL_TYPE(READ_CP)
DEF_ALL_TYPE(WRITE_CP)
DEF_ALL_TYPE(CP_ALL)
DEF_ALL_TYPE(INIT_ATT)
DEF_ALL_TYPE(GET_SIZE)

//pair<T1,T2>
template<class T1, class T2> void read_att(pair<T1,T2> &att, BufManager &bm){
	read_att(att.first,bm); read_att(att.second,bm);
}
template<class T1, class T2> void write_att(pair<T1,T2> &att, BufManager &bm) {
	write_att(att.first,bm); write_att(att.second,bm);
}
template<class T1, class T2> void init_att(pair<T1,T2> &att) {init_att(att.first); init_att(att.second);}
template<class T1, class T2> bool equ(const pair<T1,T2> &a, const pair<T1,T2> &b) {
	return equ(a.first,b.first) && equ(a.second,b.second);
}
template<class T1, class T2> void cp_att(pair<T1,T2> &a, const pair<T1,T2> &b) {
	cp_att(a.first,b.first); cp_att(a.second,b.second);
}
template<class T1, class T2> int get_size(pair<T1,T2> &att) {
	return get_size(att.first) + get_size(att.second);
}
template<class T1, class T2> bool cmp0_att(pair<T1,T2> &att) {
	return cmp0_att(att.first) && cmp0_att(att.second);
}

//vector<T>
template<class T> void read_att(vector<T> &att, BufManager &bm){
	int len; memcpy(&len, bm.buf+bm.pos, sizeof(int)); bm.pos += sizeof(int);
	att.resize(len); for(int i = 0; i < len; ++i) read_att(att[i], bm);
}
template<class T> void write_att(vector<T> &att, BufManager &bm) {
	int len = (int) att.size(); memcpy(bm.buf+bm.pos, &len, sizeof(int)); bm.pos += sizeof(int); bm.update();
	for( int i = 0; i < len; ++i) write_att(att[i], bm);
}
template<class T> void init_att(vector<T> &att) {}
template<class T> bool equ(const vector<T> &a, const vector<T> &b) {
	if(a.size() != b.size()) return false;
	for(size_t i = 0; i < a.size(); ++i) if(!equ(a[i],b[i])) return false;
	return true;
}
template<class T> void cp_att(vector<T> &a, const vector<T> &b) {
	a.resize(b.size()); for(size_t i = 0; i < b.size(); ++i) cp_att(a[i], b[i]);
}
template<class T> int get_size(vector<T> &att) {
	int s = sizeof(int);
	for(size_t i = 0; i < att.size(); ++i) s += get_size(att[i]); return s;
}
template<class T> bool cmp0_att(vector<T> &att) {return att.size()==0;}

template <class T> bool find(const vector<T> &vec, const T &val) {return find(vec.begin(),vec.end(),val) != vec.end();}
template <class T> int locate(const vector<T> &vec, const T &val) {return find(vec.begin(),vec.end(),val) - vec.begin();}
template <class T> void insert(vector<T> &v, const vector<T> &va) {v.insert(v.end(), va.begin(), va.end());}
template <class T> void insert(vector<T> &v, const T &val) {v.push_back(val);}
template <class T> void fill(vector<T> &v, const T &val) {fill(v.begin(),v.end(), val);}
template <class T> void sort(vector<T> &vec) {sort(vec.begin(),vec.end());}
template <class T> vector<T> reverse(vector<T> v){reverse(v.begin(),v.end()); return v;}
template <class T> int set_intersect(vector<T> &x, vector<T> &y, vector<T> &v) {auto it=set_intersection(x.begin(),x.end(),y.begin(),y.end(), v.begin()); return it-v.begin();}
template <class T> int set_union(vector<T> &x, vector<T> &y, vector<T> &v) {auto it=set_union(x.begin(),x.end(),y.begin(),y.end(), v.begin()); return it-v.begin();}
template <class T> int set_minus(vector<T> &x, vector<T> &y, vector<T> &v) {auto it=set_difference(x.begin(),x.end(),y.begin(),y.end(), v.begin()); return it-v.begin();}
template <class T1, class T2> void add(vector<T1> &x, vector<T1> &y, T2 c) {for(size_t i = 0; i < x.size(); ++i) x[i] += y[i]*c;}
template <class T> void add(vector<T> &x, vector<T> &y) {for(size_t i = 0; i < x.size(); ++i) x[i] += y[i];}
template <class T> T prod(vector<T> &x, vector<T> &y) {T s=0; for(size_t i = 0; i < x.size(); ++i) s+=x[i]*y[i];return s;}
template <class T1, class T2> void mult(vector<T1> &v, T2 c) {for(size_t i = 0; i < v.size(); ++i) v[i]*=c;}
template <class T> bool set_contain(vector<T> &x, vector<T> &y) { size_t lx=x.size(), ly=y.size(); if(lx<ly) return false;
	for(size_t i=0,j=0; j<y.size();){if(x[i]>y[j])return false;if(x[i]<y[j]){++i;--lx; if(lx<ly) return false;} else{++i;++j;}} return true;
}

template <class T> ostream& operator<< (std::ostream& stream, const vector<T> &x) {
	cout << "("; for(size_t i = 0; i < x.size(); ++i) if(i == 0) cout << x[i]; else cout << "," << x[i]; cout << ")"; return stream; }
template <class T1, class T2> ostream& operator<< (std::ostream& stream, const pair<T1,T2> &x) {cout << "(" << x.first << "," << x.second << ")"; return stream;}

class union_find:public vector<int> {
	public: union_find(int n) {resize(n); for(int i = 0; i < n; ++i) (*this)[i] = i;}
	public: union_find() {}
};

int get_f(int *f,int v){if(f[v]!=v) f[v]=get_f(f,f[v]); return f[v];}
void union_f(int *f, int a, int b){int fa=get_f(f,a); int fb=get_f(f,b); f[fa]=fb;}
int get_f(vector<int> &f,int v){if(f[v]!=v) f[v]=get_f(f,f[v]); return f[v];}
void union_f(vector<int> &f, int a, int b){int fa=get_f(f,a); int fb=get_f(f,b); f[fa]=fb;}

#endif