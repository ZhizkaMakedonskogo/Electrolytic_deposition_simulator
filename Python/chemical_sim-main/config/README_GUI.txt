Graphical Simulator Configuration

Files:
  config/simulator_config.json     — example config (metric, rows, n_trials, algorithm, path to histogram)

  Target histograms (example target histograms):
  config/histograms/hist1.txt
  config/histograms/hist2.txt
  config/histograms/hist3.txt

Launch:
  cd Python/chemical_sim-main
  .\.venv\Scripts\Activate.ps1
  python chemical_sim.py

Example for comparing PSO vs TPE modes:
  python scripts/pso_vs_tpe.py --targets hist1 hist2 hist3 --n-runs 5 --base-seed 42 --metric weighted_L
  python scripts/benchmark_inverse.py --methods tpe pso --swarm-size 20 --targets hist1 ...

On window startup, values from config/simulator_config.json are loaded
and the histogram from gui_defaults.theory_file is loaded automatically.

How to run optimization in the GUI:
  1) Operation mode: TPE (or PSO, GP (skopt, EI), etc.)
  2) Surface size (rows): adjustable parameter, default
  3) Metric: weighted_L
  4) Number of optimizations: adjustable parameter, default 50  (n_trials)
  5) Histogram button — hist1 is already loaded; or "Open config" select JSON
  6) Run — wait until finished
  7) JSON will be saved automatically 
  8) Result button — will additionally save the full optimizer report
     
To change the target to hist2 / hist3:
  In simulator_config.json change:
    "theory_preset": "hist2",
    "theory_file": "config/histograms/hist2.txt"
  Restart chemical_sim.py or use "Open config" → simulator_config.json

To change the metric (after the metric_contest.py competition):
  In JSON: "metric": "weighted_L",  "combined_L_new", "rsh_noise_robust"
  And select the same in the "Metric" combo box in the GUI.

Comparison of all methods ("Compare p(k) modes" button):
  Runs the headless protocol like benchmark_inverse.py
  Default parameters — from benchmark_protocol in simulator_config.json
  (n_trials=50, n_repeats=3, n_runs=5, base_seed=42).
  

Buttons:
  Open config — .json or .txt (6 lines: N, mode, b, ts, u, G)
  Save config — write the current fields to the selected .json