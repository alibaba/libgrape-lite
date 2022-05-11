#ifndef TYPE_H_
#define TYPE_H_

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

#define MAXBUF 100000000
#define BUFDLT 10000000
#define BUFEND -2
#define BUFCONT -1
#define MASTER 0

#define NONE 0
#define ONE 1
#define TWO 2
#define THREE 4
#define FOUR 8
#define FIVE 16
#define SIX 32
#define SEVEN 64
#define EIGHT 128
#define NINE 256
#define TEN 512

#define SYNALL (1<<31)

#define EPS 1e-10
#define EU -1
#define ED -2
#define ER -3
#define THRESHOLD n_vertex/50

#define PACK(...) __VA_ARGS__

#define GRAPH Graph<VTYPE>
#define Vertex VTYPE

#define VID(P) (P*n_procs+id)
#define LID(P) (P/n_procs)
#define NLOC(CID) (n/n_procs + (CID<(n%n_procs)?1:0))

#define DEF_INT_TYPE(FUNC) \
		FUNC(int) \
		FUNC(char) \
		FUNC(bool) \
		FUNC(short) \
		FUNC(long) \
		FUNC(unsigned) \
		FUNC(unsigned short) \
		FUNC(unsigned long) \
		FUNC(unsigned char) \
		FUNC(long long) \
		FUNC(unsigned long long)

#define DEF_REAL_TYPE(FUNC) \
		FUNC(float) \
		FUNC(double) \
		FUNC(long double)

#define DEF_ALL_TYPE(FUNC) \
		DEF_INT_TYPE(FUNC) \
		DEF_REAL_TYPE(FUNC)

#endif