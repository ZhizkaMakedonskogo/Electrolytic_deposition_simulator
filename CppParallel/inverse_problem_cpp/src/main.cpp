#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

void setup_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

struct Params {
    double add = 0.5;
    double transit = 0.5;
    double merge = 0.5;
};

using Histogram = std::map<int, int>;

struct EvalResult {
    double loss = std::numeric_limits<double>::infinity();
    Params params{};
};

int deterministic_seed(const int eval_index, const int rep) {
    return eval_index * 7919 + rep * 104729 + 17;
}

double clamp01(const double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

bool load_histogram_from_file(const std::string& path, Histogram& out) {
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

double weighted_hist_compare(const Histogram& theory, const Histogram& experimental) {
    if (theory.empty() || experimental.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double total_theory = 0.0;
    double total_exp = 0.0;
    for (const auto& [size, count] : theory) {
        (void)size;
        total_theory += static_cast<double>(count);
    }
    for (const auto& [size, count] : experimental) {
        (void)size;
        total_exp += static_cast<double>(count);
    }
    if (total_theory <= 0.0 || total_exp <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<int> all_sizes;
    all_sizes.reserve(theory.size() + experimental.size());
    for (const auto& [k, _] : theory) {
        all_sizes.push_back(k);
    }
    for (const auto& [k, _] : experimental) {
        all_sizes.push_back(k);
    }
    std::sort(all_sizes.begin(), all_sizes.end());
    all_sizes.erase(std::unique(all_sizes.begin(), all_sizes.end()), all_sizes.end());

    const int max_size = all_sizes.empty() ? 1 : all_sizes.back();
    double loss = 0.0;
    double weight_sum = 0.0;

    for (const int size : all_sizes) {
        const auto t_it = theory.find(size);
        const auto e_it = experimental.find(size);
        const double t_val = (t_it == theory.end() ? 0.0 : static_cast<double>(t_it->second) / total_theory);
        const double e_val = (e_it == experimental.end() ? 0.0 : static_cast<double>(e_it->second) / total_exp);
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

Histogram simulate_histogram(
    const int rows,
    const int steps,
    const Params& p,
    std::mt19937& rng
) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<int> clusters;
    clusters.reserve(static_cast<size_t>(rows));
    int free_atoms = 0;

    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < rows; ++i) {
            if (u01(rng) <= p.add) {
                if (!clusters.empty() && u01(rng) <= p.merge) {
                    std::uniform_int_distribution<int> pick(0, static_cast<int>(clusters.size()) - 1);
                    clusters[pick(rng)] += 1;
                } else {
                    free_atoms += 1;
                }
            }
        }

        if (free_atoms > 0 && !clusters.empty()) {
            int attached = 0;
            for (int i = 0; i < free_atoms; ++i) {
                if (u01(rng) <= p.transit) {
                    std::uniform_int_distribution<int> pick(0, static_cast<int>(clusters.size()) - 1);
                    clusters[pick(rng)] += 1;
                    attached += 1;
                }
            }
            free_atoms -= attached;
        }

        if (free_atoms > 0 && u01(rng) <= p.transit) {
            clusters.push_back(1);
            free_atoms -= 1;
        }

       
        if (clusters.size() >= 2) {
            std::shuffle(clusters.begin(), clusters.end(), rng);
            std::vector<int> next;
            next.reserve(clusters.size());
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (i + 1 < clusters.size() && u01(rng) <= p.merge) {
                    next.push_back(clusters[i] + clusters[i + 1]);
                    ++i;
                } else {
                    next.push_back(clusters[i]);
                }
            }
            clusters.swap(next);
        }
    }

    Histogram hist;
    for (const int size : clusters) {
        if (size > 0) {
            hist[size] += 1;
        }
    }
    if (hist.empty()) {
        hist[1] = 1;
    }
    return hist;
}

double evaluate_params_serial(
    const int rows,
    const int steps,
    const int n_repeats,
    const Params& p,
    const Histogram& theory,
    const int eval_index
) {
    double sum = 0.0;
    for (int rep = 0; rep < n_repeats; ++rep) {
        std::mt19937 rng(static_cast<unsigned int>(deterministic_seed(eval_index, rep)));
        const Histogram exp_hist = simulate_histogram(rows, steps, p, rng);
        sum += weighted_hist_compare(theory, exp_hist);
    }
    return sum / static_cast<double>(n_repeats);
}

EvalResult optimize_random_search_serial(
    const int rows,
    const int steps,
    const int n_trials,
    const int n_repeats,
    const Histogram& theory,
    const int base_seed
) {
    std::mt19937 rng(static_cast<unsigned int>(base_seed));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    EvalResult best;
    for (int t = 0; t < n_trials; ++t) {
        Params p{};
        p.add = u01(rng);
        p.transit = u01(rng);
        p.merge = u01(rng);
        const double loss = evaluate_params_serial(rows, steps, n_repeats, p, theory, t);
        if (loss < best.loss) {
            best.loss = loss;
            best.params = p;
        }
    }
    return best;
}

EvalResult optimize_random_search_parallel(
    const int rows,
    const int steps,
    const int n_trials,
    const int n_repeats,
    const Histogram& theory,
    const int base_seed,
    const int num_threads
) {
    std::atomic<int> trial_idx{0};
    std::mutex best_mutex;
    EvalResult best;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));

    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::mt19937 rng(static_cast<unsigned int>(base_seed + 97 * tid));
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            while (true) {
                const int t = trial_idx.fetch_add(1);
                if (t >= n_trials) {
                    break;
                }
                Params p{};
                p.add = clamp01(u01(rng));
                p.transit = clamp01(u01(rng));
                p.merge = clamp01(u01(rng));
                const double loss = evaluate_params_serial(rows, steps, n_repeats, p, theory, t);
                {
                    std::lock_guard<std::mutex> lock(best_mutex);
                    if (loss < best.loss) {
                        best.loss = loss;
                        best.params = p;
                    }
                }
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
    return best;
}

int main(int argc, char** argv) {
    setup_console_utf8();
    std::string target_path = "target_histogram.txt";
    if (argc > 1) {
        target_path = argv[1];
    }

    Histogram theory;
    if (!load_histogram_from_file(target_path, theory)) {
        std::cerr << "Failed to load the target histogram from: " << target_path << "\n";
        std::cerr << "File format: each line contains a pair <cluster_size> <count>\n";
        return 1;
    }

    const int rows = 64;
    const int steps = 200;
    const int n_trials = 1000;
    const int n_repeats = 4;
    const int base_seed = 42;
    const unsigned int hc = std::thread::hardware_concurrency();
    const int num_threads = hc > 0 ? static_cast<int>(hc) : 4;

    const auto t0 = std::chrono::steady_clock::now();
    const EvalResult serial_best = optimize_random_search_serial(rows, steps, n_trials, n_repeats, theory, base_seed);
    const auto t1 = std::chrono::steady_clock::now();
    const EvalResult parallel_best =
        optimize_random_search_parallel(rows, steps, n_trials, n_repeats, theory, base_seed, num_threads);
    const auto t2 = std::chrono::steady_clock::now();

    const double serial_sec = std::chrono::duration<double>(t1 - t0).count();
    const double parallel_sec = std::chrono::duration<double>(t2 - t1).count();
    const double speedup = parallel_sec > 0.0 ? serial_sec / parallel_sec : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Target histogram: " << target_path << "\n";
    std::cout << "rows=" << rows << ", steps=" << steps << ", n_trials=" << n_trials
              << ", n_repeats=" << n_repeats << ", threads=" << num_threads << "\n\n";

    std::cout << "[SERIAL]\n";
    std::cout << "best_loss=" << serial_best.loss << "\n";
    std::cout << "best_params(add, transit, merge)=(" << serial_best.params.add << ", " << serial_best.params.transit
              << ", " << serial_best.params.merge << ")\n";
    std::cout << "time_sec=" << serial_sec << "\n\n";

    std::cout << "[PARALLEL]\n";
    std::cout << "best_loss=" << parallel_best.loss << "\n";
    std::cout << "best_params(add, transit, merge)=(" << parallel_best.params.add << ", " << parallel_best.params.transit
              << ", " << parallel_best.params.merge << ")\n";
    std::cout << "time_sec=" << parallel_sec << "\n\n";

    std::cout << "speedup(serial/parallel)=" << speedup << "x\n";
    return 0;
}