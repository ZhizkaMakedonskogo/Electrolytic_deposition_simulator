#include "chemical_sim/metrics.hpp"
#include "chemical_sim/rng.hpp"
#include "chemical_sim/simulation_objective.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void setup_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

struct EvalResult {
    double loss = std::numeric_limits<double>::infinity();
    double add = 0.0;
    double transit = 0.0;
    double merge = 0.0;
};

double clamp01(const double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

EvalResult optimize_random_search_serial(
    const chemical_sim::SimulationObjective& objective,
    const int n_trials,
    const int n_repeats,
    const int steps,
    const unsigned base_seed,
    const int repeat_worker_cap
) {
    chemical_sim::Rng rng(base_seed);
    EvalResult best;
    for (int t = 0; t < n_trials; ++t) {
        const double add = rng.uniform01();
        const double transit = rng.uniform01();
        const double merge = rng.uniform01();
        const auto [loss, _] = objective.evaluate(add, transit, merge, t, n_repeats, steps, repeat_worker_cap);
        (void)_;
        if (loss < best.loss) {
            best.loss = loss;
            best.add = add;
            best.transit = transit;
            best.merge = merge;
        }
    }
    return best;
}

EvalResult optimize_random_search_parallel(
    const chemical_sim::SimulationObjective& objective,
    const int n_trials,
    const int n_repeats,
    const int steps,
    const unsigned base_seed,
    const int num_threads,
    const int repeat_worker_cap
) {
    std::atomic<int> trial_idx{0};
    std::mutex best_mutex;
    EvalResult best;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));

    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            chemical_sim::Rng rng(base_seed + 97U * static_cast<unsigned>(tid));
            while (true) {
                const int t = trial_idx.fetch_add(1);
                if (t >= n_trials) {
                    break;
                }
                const double add = clamp01(rng.uniform01());
                const double transit = clamp01(rng.uniform01());
                const double merge = clamp01(rng.uniform01());
                const auto [loss, _] =
                    objective.evaluate(add, transit, merge, t, n_repeats, steps, repeat_worker_cap);
                (void)_;
                std::lock_guard<std::mutex> lock(best_mutex);
                if (loss < best.loss) {
                    best.loss = loss;
                    best.add = add;
                    best.transit = transit;
                    best.merge = merge;
                }
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
    return best;
}

void print_usage() {
    std::cout
        << "Использование: chemical_sim_cli [theory.txt] [параметры]\n"
        << "  theory.txt     строки: <Cluster_merge> <count> (как в Python-бенчмарках)\n"
        << "  --rows N       число строк решётки (по умолчанию 64)\n"
        << "  --steps N      шагов Board.run на оценку (по умолчанию 200)\n"
        << "  --trials N     число испытаний случайного поиска (по умолчанию 200)\n"
        << "  --repeats N    число повторов n_repeats (по умолчанию 3)\n"
        << "  --repeat-workers N         лимит повторов в ПОСЛЕД. (0=послед.; без флага=auto min(n, HW))\n"
        << "  --parallel-repeat-workers N повторы в ПАРАЛЛ. (0=S2, >0=S3)\n"
        << "  --threads N                потоков для испытаний (по умолчанию: по числу ядер)\n"
        << "  --skip-serial              только блок [PARALLEL]\n";
}

int parse_int_arg(const std::vector<std::string>& args, const char* flag, int default_value) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) {
            return std::atoi(args[i + 1].c_str());
        }
    }
    return default_value;
}

bool parse_flag_present(const std::vector<std::string>& args, const char* flag) {
    for (const auto& a : args) {
        if (a == flag) {
            return true;
        }
    }
    return false;
}

int resolve_repeat_workers(const std::vector<std::string>& args, const int n_repeats) {
    if (!parse_flag_present(args, "--repeat-workers")) {
        const unsigned hc = std::thread::hardware_concurrency();
        const int h = hc > 0 ? static_cast<int>(hc) : 4;
        return std::min(n_repeats, std::max(1, h));
    }
    return parse_int_arg(args, "--repeat-workers", 0);
}

int resolve_parallel_repeat_workers(const std::vector<std::string>& args) {
    if (!parse_flag_present(args, "--parallel-repeat-workers")) {
        return 0;
    }
    return parse_int_arg(args, "--parallel-repeat-workers", 0);
}

}  // пространство имён

int main(int argc, char** argv) {
    setup_console_utf8();
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    std::string target_path = "target_histogram.txt";
    if (argc > 1 && args[1].rfind("--", 0) != 0) {
        target_path = args[1];
    }

    if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
        print_usage();
        return 0;
    }

    std::map<int, int> theory;
    if (!chemical_sim::load_histogram_from_file(target_path, theory)) {
        std::cerr << "Не удалось загрузить целевую гистограмму из: " << target_path << "\n";
        print_usage();
        return 1;
    }

    const int rows = parse_int_arg(args, "--rows", 64);
    const int steps = parse_int_arg(args, "--steps", chemical_sim::kStepsPerTrial);
    const int n_trials = parse_int_arg(args, "--trials", 200);
    const int n_repeats = parse_int_arg(args, "--repeats", chemical_sim::kDefaultNRepeats);
    int num_threads = parse_int_arg(args, "--threads", 0);
    const unsigned int hc = std::thread::hardware_concurrency();
    if (num_threads <= 0) {
        num_threads = hc > 0 ? static_cast<int>(hc) : 4;
    }

    const bool skip_serial = parse_flag_present(args, "--skip-serial");
    const int repeat_workers_serial =
        skip_serial ? 0 : resolve_repeat_workers(args, n_repeats);
    const int repeat_workers_parallel = resolve_parallel_repeat_workers(args);

    const unsigned base_seed = 42;
    const chemical_sim::SimulationObjective objective(rows, theory);

    EvalResult serial_best{};
    double serial_sec = 0.0;
    if (!skip_serial) {
        const auto t0 = std::chrono::steady_clock::now();
        serial_best =
            optimize_random_search_serial(objective, n_trials, n_repeats, steps, base_seed, repeat_workers_serial);
        const auto t1 = std::chrono::steady_clock::now();
        serial_sec = std::chrono::duration<double>(t1 - t0).count();
    }

    const auto t_par0 = std::chrono::steady_clock::now();
    const EvalResult parallel_best = optimize_random_search_parallel(
        objective, n_trials, n_repeats, steps, base_seed, num_threads, repeat_workers_parallel);
    const auto t_par1 = std::chrono::steady_clock::now();
    const double parallel_sec = std::chrono::duration<double>(t_par1 - t_par0).count();
    const double speedup = parallel_sec > 0.0 ? serial_sec / parallel_sec : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ChemicalSimCpp — полное ядро Board (C++17)\n";
    std::cout << "Целевая гистограмма: " << target_path << "\n";
    std::cout << "rows=" << rows << ", steps=" << steps << ", n_trials=" << n_trials
              << ", n_repeats=" << n_repeats << ", threads=" << num_threads
              << ", repeat_workers(serial)=" << repeat_workers_serial
              << ", parallel_repeat_workers=" << repeat_workers_parallel << "\n\n";

    if (!skip_serial) {
        std::cout << "[ПОСЛЕДОВАТЕЛЬНЫЙ случайный поиск]\n";
        std::cout << "best_loss=" << serial_best.loss << "\n";
        std::cout << "best_params(add, transit, merge)=(" << serial_best.add << ", " << serial_best.transit << ", "
                  << serial_best.merge << ")\n";
        std::cout << "time_sec=" << serial_sec << "\n\n";
    }

    std::cout << "[ПАРАЛЛЕЛЬНЫЙ случайный поиск]\n";
    std::cout << "best_loss=" << parallel_best.loss << "\n";
    std::cout << "best_params(add, transit, merge)=(" << parallel_best.add << ", " << parallel_best.transit << ", "
              << parallel_best.merge << ")\n";
    std::cout << "time_sec=" << parallel_sec << "\n\n";

    if (!skip_serial) {
        std::cout << "speedup(serial/parallel)=" << speedup << "x\n";
    }
    return 0;
}
