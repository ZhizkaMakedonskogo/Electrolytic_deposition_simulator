#pragma once

#include "board.hpp"
#include "metrics.hpp"
#include "rng.hpp"

#include <map>
#include <utility>

namespace chemical_sim {

inline constexpr int kStepsPerTrial = 200;
inline constexpr int kDefaultNRepeats = 3;

class SimulationObjective {
public:
    SimulationObjective(int rows, std::map<int, int> theory);

    std::pair<double, std::map<int, int>> run_one_repeat(
        double add,
        double transit,
        double merge,
        int eval_index,
        int rep,
        int steps = kStepsPerTrial
    ) const;

    /// repeat_worker_cap: 0 = последовательные повторы; >0 = до min(n_repeats, repeat_worker_cap) потоков.
    /// Параллельные повторы: отдельные Board/Rng на каждый повтор; ID кластеров используют std::atomic.
    std::pair<double, std::map<int, int>> evaluate(
        double add,
        double transit,
        double merge,
        int eval_index,
        int n_repeats = kDefaultNRepeats,
        int steps = kStepsPerTrial,
        int repeat_worker_cap = 0
    ) const;

private:
    int rows_;
    std::map<int, int> theory_;
};

}  // пространство имён chemical_sim
