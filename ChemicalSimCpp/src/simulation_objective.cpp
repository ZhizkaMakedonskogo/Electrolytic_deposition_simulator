#include "chemical_sim/simulation_objective.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace chemical_sim {

SimulationObjective::SimulationObjective(int rows, std::map<int, int> theory)
    : rows_(rows), theory_(std::move(theory)) {}

std::pair<double, std::map<int, int>> SimulationObjective::run_one_repeat(
    const double add,
    const double transit,
    const double merge,
    const int eval_index,
    const int rep,
    const int steps
) const {
    if (steps <= 0) {
        throw std::invalid_argument("число шагов должно быть >= 1");
    }

    Rng rng(static_cast<unsigned>(deterministic_seed(eval_index, rep)));
    Board board(rows_, add, transit, merge, Mode::Const, rng);

    for (int s = 0; s < steps; ++s) {
        board.run();
    }

    const std::map<int, int> experimental_bar = board.create_bar();
    const double loss = weighted_hist_compare(theory_, experimental_bar);
    return {loss, experimental_bar};
}

std::pair<double, std::map<int, int>> SimulationObjective::evaluate(
    const double add,
    const double transit,
    const double merge,
    const int eval_index,
    const int n_repeats,
    const int steps,
    const int repeat_worker_cap
) const {
    if (n_repeats <= 0) {
        throw std::invalid_argument("число повторов должно быть >= 1");
    }
    if (steps <= 0) {
        throw std::invalid_argument("число шагов должно быть >= 1");
    }

    std::vector<double> losses(static_cast<size_t>(n_repeats));
    std::vector<std::map<int, int>> hists(static_cast<size_t>(n_repeats));

    if (repeat_worker_cap <= 1 || n_repeats <= 1) {
        for (int rep = 0; rep < n_repeats; ++rep) {
            auto [loss, hist] = run_one_repeat(add, transit, merge, eval_index, rep, steps);
            losses[static_cast<size_t>(rep)] = loss;
            hists[static_cast<size_t>(rep)] = std::move(hist);
        }
    } else {
        const int workers = std::min(n_repeats, repeat_worker_cap);
        std::atomic<int> next_rep{0};
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(workers));

        for (int w = 0; w < workers; ++w) {
            threads.emplace_back([&, w]() {
                (void)w;
                while (true) {
                    const int rep = next_rep.fetch_add(1);
                    if (rep >= n_repeats) {
                        break;
                    }
                    auto pr = run_one_repeat(add, transit, merge, eval_index, rep, steps);
                    losses[static_cast<size_t>(rep)] = pr.first;
                    hists[static_cast<size_t>(rep)] = std::move(pr.second);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
    }

    double sum = 0.0;
    int best_i = 0;
    for (int i = 0; i < n_repeats; ++i) {
        sum += losses[static_cast<size_t>(i)];
        if (losses[static_cast<size_t>(i)] < losses[static_cast<size_t>(best_i)]) {
            best_i = i;
        }
    }
    const double mean_loss = sum / static_cast<double>(n_repeats);
    return {mean_loss, hists[static_cast<size_t>(best_i)]};
}

}  
