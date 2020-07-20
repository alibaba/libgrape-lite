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

#include <inttypes.h>
#include <stdio.h>

#include <map>
#include <unordered_map>

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: ./wcc_check <result-1> <result-2>\n");
    return 0;
  }

  std::map<uint64_t, int> cluster_name1, cluster_name2;

  FILE* fin1 = fopen(argv[1], "r");
  FILE* fin2 = fopen(argv[2], "r");

  uint64_t vid1, cid1, vid2, cid2;
  int ret = 0;

  while (fscanf(fin1, "%" PRIu64 "%" PRIu64, &vid1, &cid1) != EOF) {
    if (fscanf(fin2, "%" PRIu64 "%" PRIu64, &vid2, &cid2) == EOF) {
      printf("Vertex number not match...\n");
      ret = 1;
      break;
    }

    if (vid1 != vid2) {
      printf("Vertex id not match: %" PRIu64 " v.s. %" PRIu64 "\n", vid1, vid2);
      ret = 1;
      break;
    }

    auto iter1 = cluster_name1.find(cid1);
    auto iter2 = cluster_name2.find(cid2);

    if (iter1 == cluster_name1.end() && iter2 == cluster_name2.end()) {
      int new_cname = cluster_name1.size();
      cluster_name1[cid1] = new_cname;
      cluster_name2[cid2] = new_cname;
    } else if (iter1 != cluster_name1.end() && iter2 != cluster_name2.end()) {
      if (iter1->second != iter2->second) {
        printf("Vertex cluster name not match - A: %" PRIu64 " v.s. %" PRIu64
               "\n",
               vid1, vid2);
        ret = 1;
        break;
      }
    } else {
      printf("Vertex cluster name not match - B: %" PRIu64 " v.s. %" PRIu64
             "\n",
             vid1, vid2);
      ret = 1;
      break;
    }
  }

  fclose(fin1);
  fclose(fin2);

  return ret;
}
