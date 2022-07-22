#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

void parse_input_args(int argc, char** argv, std::string& input,
                      std::string& output) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-input")) {
      input = std::string(argv[++i]);
      continue;
    }

    if (!strcmp(argv[i], "-output")) {
      output = std::string(argv[++i]);
      continue;
    }
  }

  if (input.empty() || output.empty() || access(input.c_str(), 0) != 0) {
    std::cerr << "Invalid param";
    exit(1);
  }
}

void assert_msg(bool b, const std::string& msg) {
  if (!b) {
    std::cerr << msg << std::endl;
    exit(1);
  }
}

template <typename W_T>
struct Edge {
  Edge(size_t nbr_, W_T data_) : nbr(nbr_), data(data_) {}
  size_t nbr;
  W_T data;
};

namespace std {
template <typename T>
struct hash<Edge<T>> {
  std::size_t operator()(const Edge<T>& e) const { return e.nbr; }
};

}  // namespace std

int main(int argc, char* argv[]) {
  std::string input, output;

  parse_input_args(argc, argv, input, output);

  std::ifstream infile(input);
  std::string line;
  size_t line_no = 0;
  size_t nv, ne;
  bool weighted = false;
  using weight_t = int;

  std::vector<std::unordered_map<size_t, weight_t>> graph;
  size_t block_size = 0;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);

    if (line[0] == '%') {
      continue;
    }

    line_no++;

    if (line_no == 1) {
      iss >> nv >> nv >> ne;
      block_size = ne * 0.01 + 1;
      std::cout << "NV: " << nv << " NE: " << ne << std::endl;
      graph.resize(nv + 1);
    } else {
      size_t u, v;
      iss >> u >> v;

      assert_msg(u >= 1 && u <= nv && v >= 1 && v <= nv,
                 "Invalid line:" + line);

      weight_t w;

      if (iss >> w) {
        weighted = true;
      }

      graph[u].emplace(v, w);
    }

    if (line_no % block_size == 0) {
      std::cout << "Progress: " << 1.0 * line_no / ne * 100 << " %"
                << std::endl;
    }
  }

  infile.close();

  std::cout << "Inserting reversed edges" << std::endl;

  for (size_t u = 1; u <= nv; u++) {
    auto& edges = graph[u];
    for (auto edge : edges) {
      auto v = edge.first;
      auto w = edge.second;

      if (u != v && graph[v].count(u) == 0) {
        graph[v].emplace(u, w);
      }
    }
  }

  std::cout << "Writing" << std::endl;
  std::ofstream outfile(output.c_str());

  line_no = 0;
  ne = 0;
  for (auto& edges : graph) {
    ne += edges.size();
  }
  block_size = ne * 0.01 + 1;
  outfile << nv << " " << nv << " " << ne << std::endl;

  for (size_t u = 1; u <= nv; u++) {
    auto& edges = graph[u];

    for (auto edge : edges) {
      outfile << u << " " << edge.first;
      if (weighted) {
        outfile << " " << edge.second;
      }

      outfile << "\n";

      line_no++;
      if (line_no % block_size == 0) {
        std::cout << "Progress: " << 1.0 * line_no / ne * 100 << " %"
                  << std::endl;
        outfile.flush();
      }
    }
  }
  outfile.close();
}