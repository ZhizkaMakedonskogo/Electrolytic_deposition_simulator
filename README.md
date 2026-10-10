# Electrolytic Deposition Simulator and Inverse Parameter Identification Problem
## About the Project

Direct physical selection of deposition modes is expensive and scales poorly. The model describes stochastic dynamics on a lattice of the active surface: the appearance of an atom, the transition and merging of clusters. The output of the forward problem is a size distribution histogram.
The inverse problem is formulated as minimizing a noisy black-box loss function: the target histogram is compared with the averaged result of several simulator runs (`n_repeats`). A unified `SimulationObjective` module and the `weighted_hist_compare` metric are used so that the comparison of methods is correct under a single protocol (evaluation budget, seeds, surface size, set of targets).

## What Has Been Implemented

 Simulation model (`Board`, `Cluster`, `Cell`) and a graphical simulator on tkinter
 Inverse problem with averaging over repeats and switching of loss metrics
 A set of optimizers: TPE (Optuna), GP / EI, skopt, PSO, GA, differential evolution, CMA-ES, trust-region BO, multi-fidelity, etc.
 Scripts for some comparative runs
 C++17 core of the model with parallel execution

## Repository Structure

 chemical_sim-main - GUI, model, optimizers, configs, and target histograms
 ChemicalSimCpp - Parallel C++ implementation of the model and CLI random search
 CppParallel Compact C++ benchmark for the inverse problem

## Requirements

 Python 3.10+
 CMake 3.16+ and a C++17 compiler (MSVC, GCC, or Clang)

## Quick Start (Python, GUI)

Windows (PowerShell):

cd Python/chemical_sim-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python chemical_sim.py

Linux / macOS:

cd Python/chemical_sim-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python chemical_sim.py


## Building C++

Full core:
cd ChemicalSimCpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\chemical_sim_cli.exe ..\CppParallel\inverse_problem_cpp\target_histogram.txt --trials 50 --repeat-workers 4

Compact benchmark:
cd CppParallel/inverse_problem_cpp
cmake -S . -B build
cmake --build build --config Release
.\build\Release\inverse_problem.exe
