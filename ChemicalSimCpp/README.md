# ChemicalSimCpp

A parallel C++17 implementation of the simulation model and the core of the inverse problem (Board, Cluster, Cell, SimulationObjective.weighted_hist_compare, deterministic seed).

## What's Inside

 Forward run (Board::run, create_bar)
 The weighted_hist_compare metric and loading of the target histogram from a file (as in CppParallel/inverse_problem_cpp).
 The SimulationObjective class: several repeats n_repeats, evaluation index, and repeat noise.
 The chemical_sim_cli console program: sequential and parallel random search in [0, 1]^3, output of the best solution and speedup.

## Parallelism

 By trial (main.cpp): a pool of std::thread, std::atomic<int> for the trial counter, std::mutex for updating best.
 By rep (SimulationObjective::evaluate), where with repeat_workers > 1 (by default min(n_repeats, hardware_concurrency()) in parallel random search) — a separate pool, writing to losses[rep] without a mutex.
 In sequential random search, inside evaluate, repeat_worker_cap = 0 is set so as not to launch two pools at once

repeat_workers 0 — all repeats in a single thread.

## What Is Not Ported from Python

The GUI (tkinter), Optuna, skopt, and the remaining optimizers stay in Python.

## Build

cd ChemicalSimCpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

Example run (paths relative to the repository root):

bash
./build/chemical_sim_cli ../CppParallel/inverse_problem_cpp/target_histogram.txt --rows 64 --trials 100 --repeats 3

PowerShell:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\chemical_sim_cli.exe ..\CppParallel\inverse_problem_cpp\target_histogram.txt --trials 50 --repeat-workers 4


## Structure

 include/chemical_sim/ — headers
 src/ — implementation