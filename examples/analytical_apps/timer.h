/**
 * NOLINT(legal/copyright)
 *
 * The file examples/analytical_apps/timer.h is referred and derived from
 * project atlarge-research/graphalytics-platforms-powergraph,
 *
 *    https://github.com/atlarge-research/graphalytics-platforms-powergraph/
 * blob/master/src/main/c/utils.hpp
 *
 * which has the following license:
 *
 *  Copyright 2015 Delft University of Technology
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef EXAMPLES_ANALYTICAL_APPS_TIMER_H_
#define EXAMPLES_ANALYTICAL_APPS_TIMER_H_

#include <stddef.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>
#include <utility>
#include <vector>

/**
 * Timers for LDBC benchmarking, referred and derived from project
 * atlarge-research/graphalytics-platforms-powergraph.
 */
static double timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static bool timer_enabled;
static std::vector<std::pair<std::string, double>> timers;

static void timer_start(bool enabled = true) {
  timers.clear();
  timer_enabled = enabled;
}

static void timer_next(const std::string& name) {
  if (timer_enabled) {
    timers.emplace_back(std::make_pair(name, timer()));
  }
}

static void timer_end() {
  if (timer_enabled) {
    timer_next("end");

    std::cerr << "Timing results:" << std::endl;

    for (size_t i = 0; i < timers.size() - 1; i++) {
      std::string& name = timers[i].first;
      double time = timers[i + 1].second - timers[i].second;

      std::cerr << " - " << name << ": " << time << " sec" << std::endl;
      std::cerr << std::endl;
    }

    timers.clear();
  }
}

#endif  // EXAMPLES_ANALYTICAL_APPS_TIMER_H_
