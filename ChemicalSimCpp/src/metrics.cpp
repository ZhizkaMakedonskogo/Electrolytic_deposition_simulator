#include "chemical_sim/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace chemical_sim {

int deterministic_seed(const int eval_index, const int rep) {
    return eval_index * 7919 + rep * 104729 + 17;
}

double weighted_hist_compare(const std::map<int, int>& theory, const std::map<int, int>& experimental) {
    if (theory.empty() || experimental.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double total_theory = 0.0;
    double total_exp = 0.0;
    for (const auto& [_, c] : theory) {
        (void)_;
        total_theory += static_cast<double>(c);
    }
    for (const auto& [_, c] : experimental) {
        (void)_;
        total_exp += static_cast<double>(c);
    }
    if (total_theory <= 0.0 || total_exp <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    std::set<int> all_sizes;
    for (const auto& [k, _] : theory) {
        all_sizes.insert(k);
    }
    for (const auto& [k, _] : experimental) {
        all_sizes.insert(k);
    }

    const int max_size = all_sizes.empty() ? 1 : *all_sizes.rbegin();
    double loss = 0.0;
    double weight_sum = 0.0;

    for (int size : all_sizes) {
        const double t_val =
            static_cast<double>(theory.count(size) ? theory.at(size) : 0) / total_theory;
        const double e_val =
            static_cast<double>(experimental.count(size) ? experimental.at(size) : 0) / total_exp;
        const double weight = max_size > 0 ? static_cast<double>(size) / static_cast<double>(max_size) : 1.0;
        const double diff = t_val - e_val;
        loss += weight * diff * diff;
        weight_sum += weight;
    }

    if (weight_sum > 0.0) {
        loss /= weight_sum;
    }

    const double cluster_count_diff =
        static_cast<double>(std::abs(static_cast<int>(theory.size()) - static_cast<int>(experimental.size()))) /
        static_cast<double>(std::max({theory.size(), experimental.size(), static_cast<size_t>(1)}));
    loss += 0.1 * cluster_count_diff;

    return loss * 100.0;
}

bool load_histogram_from_file(const std::string& path, std::map<int, int>& out) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        int k = 0;
        int v = 0;
        if (iss >> k >> v) {
            if (k > 0 && v > 0) {
                out[k] += v;
            }
        }
    }
    return !out.empty();
}

}  