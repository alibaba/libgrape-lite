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
#include <string>

#include <cmath>

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

  FILE* fin1 = fopen(argv[1], "r");
  FILE* fin2 = fopen(argv[2], "r");

  unsigned long long id1, id2;
  char val1[128], val2[128];

  int ret = 0;

  while (fscanf(fin1, "%21llu%21s", &id1, val1) != EOF) {
    if (fscanf(fin2, "%21llu%21s", &id2, val2) == EOF) {
      printf("Vertex number not match...\n");
      ret = 1;
      break;
    }

    if (id1 != id2) {
      printf("Vertex id not match: %llu v.s. %llu\n", id1, id2);
      ret = 1;
      break;
    }

    if (!match(parse(val1), parse(val2))) {
      printf("Value of [vertex-%llu] not match: %s v.s. %s\n", id1, val1, val2);
      ret = 1;
      break;
    }
  }

  fclose(fin1);
  fclose(fin2);

  return ret;
}
