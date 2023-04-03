/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>

const double MIN_NEAR_INFINITY = std::numeric_limits<double>::max() * 0.999;
const double MAX_NEAR_ZERO = std::numeric_limits<double>::min() * 10;
const double COMPARISON_THRSHOLD = 0.0001;

double parse(const std::string& val) {
  std::string low = val;
  std::transform(low.begin(), low.end(), low.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (low == "inf" || low == "+inf" || low == "infinity" ||
      low == "+infinity") {
    return std::numeric_limits<double>::max();
  } else if (low == "-inf" || low == "-infinity") {
    return std::numeric_limits<double>::lowest();
  } else {
    return std::stod(low);
  }
}

bool is_near_infinity(double x) {
  return (x == std::numeric_limits<double>::max()) ||
         (std::fabs(x) > MIN_NEAR_INFINITY);
}

bool is_near_zero(double x) { return (std::fabs(x) < MAX_NEAR_ZERO); }

bool match(double v1, double v2) {
  if (v1 == v2) {
    return true;
  } else if (is_near_infinity(v1) && is_near_infinity(v2)) {
    return true;
  } else if (is_near_zero(v1) && is_near_zero(v2)) {
    return true;
  } else {
    return fabs(v1 - v2) < (COMPARISON_THRSHOLD * v1);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: ./eps_check <correct-result> <output-result>\n");
    return 0;
  }

  std::string fname1 = argv[1];
  std::string fname2 = argv[2];

  std::ifstream fin1(fname1);
  std::ifstream fin2(fname2);

  int64_t id1, id2;
  std::string val1, val2;

  int ret = 0;

  while (fin1 >> id1 >> val1) {
    if (!(fin2 >> id2 >> val2)) {
      printf("Vertex number not match, %s too few...", fname2.c_str());
    }
    if (id1 != id2) {
      printf("Vertex id not match: %lu v.s. %lu\n", id1, id2);
      ret = 1;
      break;
    }

    if (!match(parse(val1), parse(val2))) {
      printf("Value of [vertex-%lu] not match: %s v.s. %s\n", id1, val1.c_str(), val2.c_str());
      ret = 1;
      break;
    }
  }
  if (fin1.eof()) {
    if (fin2 >> id2) {
      printf("Vertex number not match, %s too few...", fname1.c_str());
    }
    if (!fin2.eof()) {
      printf("%s not reach end of file...", fname2.c_str());
    }
  } else {
    printf("%s not reach end of file...", fname1.c_str());
  }

  return ret;
}