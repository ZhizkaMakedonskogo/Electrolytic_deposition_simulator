#pragma once

#include <map>
#include <string>

namespace chemical_sim {

int deterministic_seed(int eval_index, int rep);

double weighted_hist_compare(const std::map<int, int>& theory, const std::map<int, int>& experimental);

bool load_histogram_from_file(const std::string& path, std::map<int, int>& out);

}  // пространство имён chemical_sim
