from classes.coordinate_descent import CoordinateDescent
from classes.default_runner import DefaultRunner
from classes.optuna_integration import OptunaIntegration
from classes.CMA_ES_4 import CmaEsOptunaIntegration
from classes.gp_bayesian_optimization import GPBayesianOptimization
from classes.skopt_gp_optimization import SkoptGpOptimization
from classes.multi_fidelity_optimizer import MultiFidelityOptimizer
from classes.xnes_optimization import XnesOptimization
from classes.pso_optimization import PsoOptimization
from classes.ga_optimization import GaOptimization
from classes.differential_evolution_optimization import DifferentialEvolutionOptimization
from classes.trust_region_bo import TrustRegionBoOptimization
from classes.turbo_mt_optimization import TurboMtOptimization
from classes.optuna_hyperband_integration import OptunaHyperbandIntegration
from classes.lhs_nelder_mead import LhsNelderMeadOptimization
from classes.rf_ei_optimization import RfEiOptimization
from classes.cem_optimization import CemOptimization
from classes.simulated_annealing_optimization import SimulatedAnnealingOptimization
from resources.utilities import run_time
from resources.constants import (
    ALGORITHMS_INVERSE_PROBLEM,
    ALGORITHMS_MODE_COMPARE_BUNDLE,
    ALGORITHMS_NEED_OPTI_OR_BUDGET,
    ALGORITHMS_USE_LOADED_HISTOGRAM,
    Algorithm,
    Mode,
    UISize,
)
from interfaces.sleeper import Sleeper

from interfaces.drawer import Drawer

import math
import tkinter as tk
from tkinter import PanedWindow, ttk, messagebox, HORIZONTAL, VERTICAL, DISABLED, NORMAL, ACTIVE
from typing import Any, List, Optional, Tuple
from tkinter.filedialog import askopenfilename, asksaveasfilename

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from classes.board import Board
from classes.simulation_objective import (
    METRIC_CHOICES,
    METRIC_COMBINED_L_NEW,
    METRIC_RSH_NOISE_ROBUST,
    METRIC_WEIGHTED_L,
    STEPS_PER_TRIAL,
    get_active_metric,
    get_active_metric_function,
    set_active_metric,
)
from resources.gui_benchmark_export import (
    BENCHMARK_PRESETS,
    benchmark_protocol_defaults,
    generate_benchmark_figures,
    list_selectable_benchmark_methods,
    mode_compare_results_from_run,
    run_mode_compare_benchmark,
)
from resources.method_benchmark_params import (
    METHOD_PARAM_SCHEMA,
    SEED_PRESETS,
    coerce_method_options,
    default_method_options,
    parse_seed_list,
)


class ChemicalAppUI(Drawer, Sleeper):  # (tk)
    def __init__(self) -> None:
        self.N = 0
        self.mode = 0
        self.b = 0.0
        self.ts = 0.0
        self.u = 0.0
        self.G = 0
        self.current_G = 0

        self.theory = None
        self.multiplier = 0.0

        self.opti_count = 0
        self.opti_budget = 100.0

        self._is_drawing = False
        self._hist_figure = None
        self._bar_resize_job = None
        self._canvas_resize_job = None

        # root must exist before DoubleVar/BooleanVar (Python 3.9+ tk)
        self.window = tk.Tk()
        self.window.geometry(f"{UISize.WIN_W.value}x{UISize.WIN_H.value}")
        self.window.minsize(800, 480)
        self.window.resizable(True, True)

        self._bar_fig_scale = tk.DoubleVar(master=self.window, value=1.0)
        self._canvas_area_scale = tk.DoubleVar(master=self.window, value=1.0)

        # Side panel
        self.outer_pane = PanedWindow(
            self.window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5, bg="#ddd"
        )
        self.triple_pane = PanedWindow(
            self.outer_pane, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5, bg="#ccc"
        )

        self.sidebar = tk.Frame(self.outer_pane, bd=1, relief=tk.GROOVE, padx=2, pady=2)
        # Top — only fields (scroll). Bottom — buttons/delay, always on screen
        self.sb_form_area = tk.Frame(self.sidebar)
        self.sb_actions = tk.Frame(self.sidebar, padx=0, pady=2)
        self._sb_inner, _sb_cv = self._pack_vertical_scroll(self.sb_form_area)
        self._sb_scroll_canv = _sb_cv

        self.graphbar = tk.Frame(self.triple_pane, bd=2, relief=tk.GROOVE)
        self.bar_top = tk.Frame(self.graphbar)
        self._graph_scrolled_area = tk.Frame(self.graphbar)
        self._graph_inner, _ = self._pack_vertical_scroll(
            self._graph_scrolled_area, fill_height=True
        )
         # Vertical divider
        self._bar_pane = PanedWindow(
            self._graph_inner,
            orient=VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            bd=0,
            bg="#e8e8e8",
        )
        self.bar_host = tk.Frame(self._bar_pane, bg="white")
        self._bar_pane_filler = tk.Frame(self._bar_pane, bg="#f5f5f5")
        self._bar_filler_hint = tk.Label(
            self._bar_pane_filler,
            text="↕ тяните разделитель",
            fg="#999999",
            bg="#f5f5f5",
            font=("TkDefaultFont", 8),
        )
        self._bar_filler_hint.pack(pady=4)
        self._bar_pane.bind("<Configure>", self._on_bar_pane_configure)

        self.canvas_frame = tk.Frame(self.triple_pane, bd=2, relief=tk.GROOVE)
        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#012", highlightthickness=0, relief=tk.FLAT, bd=0
        )
        self._sim_vsb = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self._sim_hsb = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.config(
            yscrollcommand=self._sim_vsb.set,
            xscrollcommand=self._sim_hsb.set,
        )

        self.statbar = tk.Frame(self.triple_pane, bd=2, relief=tk.GROOVE, padx=4, pady=4)
        self._stat_inner, _ = self._pack_vertical_scroll(self.statbar)

        self.vcmd = (self.window.register(self.validate), "%P")

        self.graph = None

        self.label_modelling = tk.Label(self._sb_inner, text="Operation mode")
        self.combobox_algo = ttk.Combobox(
            self._sb_inner,
            values=[
                Algorithm.DEFAULT.value,
                Algorithm.DOWNHILL.value,
                Algorithm.TPE.value,
                Algorithm.XNES.value,
                Algorithm.CMA_ES.value,
                Algorithm.GP_BAYES.value,
                Algorithm.SKOPT_GP.value,
                Algorithm.MULTI_FIDELITY.value,
                Algorithm.PSO.value,
                Algorithm.GA.value,
                Algorithm.DE.value,
                Algorithm.TRUST_REGION_BO.value,
                Algorithm.TURBO_MT.value,
                Algorithm.HYPERBAND_TPE.value,
                Algorithm.LHS_NM.value,
                Algorithm.RF_EI.value,
                Algorithm.CEM.value,
                Algorithm.SA.value,
            ],
        )
        self.combobox_algo.bind("<<ComboboxSelected>>", self.change_opti_visible)
        self.label_height = tk.Label(self._sb_inner, text="Surface size")
        self.textbox_height = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.label_mode = tk.Label(self._sb_inner, text="Simulation mode")
        self.combobox_mode = ttk.Combobox(self._sb_inner, values=[Mode.CONST.value, Mode.VAR.value])
        self.combobox_mode.bind("<<ComboboxSelected>>", self.change_label_create)
        self.label_create = tk.Label(self._sb_inner, text="Creation probability")
        self.textbox_create = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.label_ts = tk.Label(self._sb_inner, text="Transition probability")
        self.textbox_ts = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.label_margin = tk.Label(self._sb_inner, text="Merge probability")
        self.textbox_margin = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.label_count = tk.Label(self._sb_inner, text="Number of iterations")
        self.textbox_count = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.label_opti = tk.Label(self._sb_inner, text="Number of optimizations")
        self.textbox_opti = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.textbox_opti.config(state=DISABLED)
        self.label_pso_swarm = tk.Label(self._sb_inner, text="PSO: swarm size (0=auto)")
        self.textbox_pso_swarm = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.textbox_pso_swarm.insert(0, "0")
        self.label_multiplier = tk.Label(self._sb_inner, text="Initial step")
        self.textbox_multiplier = tk.Entry(self._sb_inner, validate="key", validatecommand=self.vcmd)
        self.textbox_multiplier.config(state=DISABLED)
        self.label_metric = tk.Label(self._sb_inner, text="Optimization metric")
        self.combobox_metric = ttk.Combobox(
            self._sb_inner,
            values=[
                f"{METRIC_WEIGHTED_L} (old L)",
                f"{METRIC_COMBINED_L_NEW} (0.6*sqrt(JS)+0.4*W1_norm)",
                f"{METRIC_RSH_NOISE_ROBUST} (robust W1+Hellinger)",
            ],
            state="readonly",
        )

        self._row_run = tk.Frame(self.sb_actions)
        self._col_run_l = tk.Frame(self._row_run)
        self._col_run_r = tk.Frame(self._row_run)
        self.btb_run = tk.Button(self._col_run_l, text="Run", command=self.run_btn)
        self.no_ui = tk.BooleanVar(master=self.window)
        self.no_bar = tk.BooleanVar(master=self.window)
        self.cb_run = tk.Checkbutton(self._col_run_r, text="No canvas", variable=self.no_ui)
        self.cb_bar = tk.Checkbutton(self._col_run_r, text="No bar", variable=self.no_bar)
        self.btn_pause = tk.Button(self._col_run_l, text="Pause", command=self.pause_btn)
        self.btn_restart = tk.Button(self._col_run_l, text="Rerun", command=self.restart_btn)
        self.btn_pause.config(state=DISABLED)
        self.btn_restart.config(state=DISABLED)
        self.btn_result = tk.Button(self._col_run_l, text="Result", command=self.result_btn)
        self.btn_result.config(state=DISABLED)
        self.btn_openhist = tk.Button(self._col_run_r, text="Histogram", command=self.open_hist)

        self._row_config = tk.Frame(self.sb_actions)
        self.btn_openconfig = tk.Button(self._row_config, text="Open config", command=self.open_configfile)
        self.btn_saveconfig = tk.Button(self._row_config, text="Save config", command=self.save_configfile)
        self.btn_mode_compare = tk.Button(
            self.sb_actions,
            text="Benchmark: select methods…",
            command=self._open_mode_compare_dialog,
        )
        self._compare_modes_busy = False

        # Histogram/canvas scale
        self._bar_scales = tk.Frame(self.sb_actions)
        self.label_scale_bar = tk.Label(self._bar_scales, text="Bar (fig)")
        self.scale_bar_mpl = tk.Scale(
            self._bar_scales,
            from_=0.5,
            to=2.0,
            orient=HORIZONTAL,
            resolution=0.05,
            variable=self._bar_fig_scale,
            command=self._on_bar_mpl_scale,
            length=100,
        )
        self.label_scale_cv = tk.Label(self._bar_scales, text="Grid")
        self.scale_cv_points = tk.Scale(
            self._bar_scales,
            from_=0.4,
            to=1.5,
            orient=HORIZONTAL,
            resolution=0.05,
            variable=self._canvas_area_scale,
            length=100,
        )

        self._var_sleep = tk.DoubleVar(master=self.window, value=0.0)
        self.label_sleep = tk.Label(self.sb_actions, text="Delay between iterations")
        self._label_sleep_value = tk.Label(self.sb_actions, text="0.0")
        self.scale_sleep = tk.Scale(
            self.sb_actions,
            from_=0,
            to=3,
            orient=HORIZONTAL,
            resolution=0.1,
            length=120,
            variable=self._var_sleep,
            command=self._on_sleep_display,
        )

        self.btn_save_hist = tk.Button(self.bar_top, text="Сохранить bar", command=self.save_histogram)

        self.stat_1 = tk.Label(self._stat_inner)
        self.stat_2 = tk.Label(self._stat_inner)
        self.stat_3 = tk.Label(self._stat_inner)
        self.stat_compare = tk.Label(self._stat_inner, justify="left", wraplength=320)

        self._method = None
        self._method_status = False
        self.is_exit = False

    def _pack_vertical_scroll(
        self, parent: tk.Widget, *, fill_height: bool = False
    ) -> Tuple[tk.Frame, tk.Canvas]:
        """Section with vertical scroll (Canvas + Scrollbar). The inner frame by default
        stretches to the canvas width; with fill_height=True — also to the height (needed for
        the histogram: otherwise the inner frame does not get the viewport height and the plot is "squashed")"""
        try:
            bg = parent.cget("bg")
        except tk.TclError:
            bg = None
        cv = tk.Canvas(parent, highlightthickness=0, borderwidth=0, bg=bg)
        vsb = tk.Scrollbar(parent, orient=tk.VERTICAL, command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        inner = tk.Frame(cv, bg=bg)
        win_id = cv.create_window(0, 0, window=inner, anchor=tk.NW)

        def _inner_cfg(_: tk.Event) -> None:
            cv.configure(scrollregion=cv.bbox("all"))

        def _cv_cfg(e: tk.Event, wid: int = win_id) -> None:
            w = int(e.width)
            h = int(e.height)
            if w <= 2:
                return
            try:
                if fill_height and h > 2:
                    cv.itemconfigure(wid, width=w, height=h)
                else:
                    cv.itemconfigure(wid, width=w)
            except tk.TclError:
                pass

        inner.bind("<Configure>", _inner_cfg)
        cv.bind("<Configure>", _cv_cfg)

        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _wheel_v(e, sc=cv) -> None:
            if hasattr(e, "delta") and e.delta:
                sc.yview_scroll(int(-1 * (e.delta / 120)), "units")
            else:
                n = int(getattr(e, "num", 0) or 0)
                if n == 4:
                    sc.yview_scroll(-2, "units")
                elif n == 5:
                    sc.yview_scroll(2, "units")

        cv.bind("<MouseWheel>", _wheel_v)
        inner.bind("<MouseWheel>", _wheel_v)
        for b in (4, 5):
            inner.bind(f"<Button-{b}>", _wheel_v)
            cv.bind(f"<Button-{b}>", _wheel_v)

        return inner, cv

    def _scroll_sim_canvas(self, e) -> None:
        c = self.canvas
        st = int(getattr(e, "state", 0) or 0)
        shift = bool(st & 0x0001)
        if hasattr(e, "delta") and e.delta:
            if shift:
                c.xview_scroll(int(-1 * (e.delta / 120)), "units")
            else:
                c.yview_scroll(int(-1 * (e.delta / 120)), "units")
        else:
            n = int(getattr(e, "num", 0) or 0)
            d = -2 if n == 4 else 2 if n == 5 else 0
            if d == 0:
                return
            if shift:
                c.xview_scroll(d, "units")
            else:
                c.yview_scroll(d, "units")

    def _refresh_canvas_scrollregion(self) -> None:
        try:
            self.canvas.update_idletasks()
            bb = self.canvas.bbox("all")
            if bb:
                self.canvas.configure(scrollregion=bb)
        except Exception:
            pass

    def _on_sleep_display(self, value: str) -> None:
        try:
            self._label_sleep_value.config(text=f"{float(value):.1f}")
        except (ValueError, tk.TclError):
            pass

    def open_configfile(self) -> None:
        filepath = askopenfilename(
            filetypes=[
                ("Config", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if not filepath:
            return
        if filepath.lower().endswith(".json"):
            try:
                from resources.simulator_config import apply_gui_defaults, load_config_file, save_config

                cfg = load_config_file(filepath)
                apply_gui_defaults(self, cfg)
                save_config(cfg)
                self.is_exit = False
                self.window.title(f"Chemical Simulator - {filepath}")
                messagebox.showinfo(
                    "Config",
                    "Loaded simulator_config.json:\n"
                    "GUI fields, metric, histogram (theory_file).\n"
                    "After Run, the JSON will be saved to results",
                )
           except Exception as e:
                messagebox.showerror("Config", f"Error reading JSON:\n{e}")
            return
        with open(filepath, "r") as input_file:
            text = input_file.readlines()
            if len(text) < 6:
                messagebox.showinfo(
                    "Warning",
                    "Invalid config format: at least 6 lines with one value per line are required:\n"
                    "N, mode (0 or 1), b, ts, u, G.\n"
                    f"Current number of lines in the file: {len(text)}.\n"
                    "Load the theory (histogram) using the Histogram button.",
                )
                self.is_exit = True
                return
            else:
                self.is_exit = False
                self.fields_clearing()
                self.textbox_height.insert(0, text[0].replace('\n', ''))
                self.combobox_mode.insert(0, Mode.VAR.value if text[1].replace('\n', '') == '1' else Mode.CONST.value)
                self.textbox_create.insert(0, text[2].replace('\n', ''))
                self.textbox_ts.insert(0, text[3].replace('\n', ''))
                self.textbox_margin.insert(0, text[4].replace('\n', ''))
                self.textbox_count.insert(0, text[5].replace('\n', ''))

        self.window.title(f"Chemical Simulator - {filepath}")

    def save_configfile(self) -> None:
        filepath = asksaveasfilename(
            defaultextension="txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        )
        if not filepath:
            return
        with open(filepath, 'w') as output_file:
            self.set_params()
            for item in [self.N, self.mode, self.b, self.ts, self.u, self.current_G]:
                output_file.write(str(item) + '\n')
        self.window.title(f"Chemical Simulator - {filepath}")

    def _parse_theory_line_string(self, raw: str) -> dict:
        """Parses 'k:v,k2:v2' into a dictionary; skips empty segments."""
        text = raw.strip().replace("{", "").replace("}", "")
        if not text:
            return {}
        out = {}
        float_keys = False
        for point in text.replace(" ", "").split(","):
            if not point or ":" not in point:
                continue
            a, b = point.split(":", 1)
            key = float(a)
            val = float(b)
            if key != int(key):
                float_keys = True
            out[key] = val
        if not out:
            return {}
        if not float_keys:
            out = {int(k): v for k, v in out.items()}
        return out

   def open_hist(self) -> None:
        """Load the theoretical histogram (one or more lines in the file)"""
        filepath = askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, "rb") as bf:
                raw = bf.read()
        except OSError as e:
            messagebox.showerror("Histogram", f"Failed to read the file:\n{e}")
            return
        full = None
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                full = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if full is None:
            full = raw.decode("utf-8", errors="replace")

        joined = ",".join(
            ln.strip() for ln in full.splitlines() if ln.strip() and not ln.strip().startswith("#")
        )
comma_blob = full.replace("\r\n", ",").replace("\n", ",").replace("\r", ",")
        for candidate in (joined, comma_blob, full.replace("\ufeff", "").strip()):
            if not candidate:
                continue
            try:
                self.theory = self._parse_theory_line_string(candidate)
            except (ValueError, TypeError) as e:
                print(f"parse attempt: {e}")
                self.theory = None
                continue
            if self.theory:
                n = len(self.theory)
                kmin, kmax = min(self.theory.keys()), max(self.theory.keys())
                print(f"Theoretical histogram: {n} values, keys {kmin}…{kmax}")
                messagebox.showinfo(
                    "Histogram",
                    f"Loaded theoretical distribution: {n} points.\n" f"File: {filepath}",
                )
                return
        self.theory = None
        messagebox.showerror(
            "Histogram",
            "Failed to parse the file. The format must be: key:value separated by commas, one or more lines.\n"
            "Example: 0:0.1,1:0.2,2:0.3",
        )

    @staticmethod
    def validate(new_value) -> bool:
        try:
            if new_value == '' or new_value == '-' or new_value == '+':
                return True
            _str = str(float(new_value))
            return True
        except:
            return False

    def set_params(self) -> None:
        algo = self.combobox_algo.get()
        metric_raw = (self.combobox_metric.get() or "").strip()
        metric_name = metric_raw.split(" ", 1)[0] if metric_raw else METRIC_WEIGHTED_L
        set_active_metric(metric_name if metric_name in METRIC_CHOICES else METRIC_WEIGHTED_L)
        try:
            from resources.simulator_config import save_config

            save_config({"metric": get_active_metric()})
            if hasattr(self, "textbox_pso_swarm"):
                try:
                    sw = int(float((self.textbox_pso_swarm.get() or "0").replace(",", ".")))
                except (TypeError, ValueError):
                    sw = 0
                save_config({"gui_defaults": {"pso_swarm_size": sw}})
        except Exception as _cfg_err:
            print(f"simulator_config.json not updated: {_cfg_err}")

        # Check required fields
        missing_fields = []

        if algo == '' and self.combobox_algo["state"] == NORMAL:
            missing_fields.append("Operation mode")
        if self.textbox_height["state"] == NORMAL and self.textbox_height.get().strip() == '':
            missing_fields.append("Surface size")
        if self.combobox_mode["state"] == NORMAL and self.combobox_mode.get().strip() == '':
            missing_fields.append("Simulation mode")

        # For modes requiring probabilities
        if algo not in ALGORITHMS_INVERSE_PROBLEM:
            if self.textbox_create["state"] == NORMAL and self.textbox_create.get().strip() == '':
                missing_fields.append("Creation probability")
            if self.textbox_ts["state"] == NORMAL and self.textbox_ts.get().strip() == '':
                missing_fields.append("Transition probability")
            if self.textbox_margin["state"] == NORMAL and self.textbox_margin.get().strip() == '':
                missing_fields.append("Merge probability")

        # Required fields for all modes
        if self.textbox_count["state"] == NORMAL and self.textbox_count.get().strip() == '':
            missing_fields.append("Number of iterations")

        # Fields specific to optimization algorithms
        if algo in ALGORITHMS_NEED_OPTI_OR_BUDGET:
            if self.textbox_opti["state"] == NORMAL and self.textbox_opti.get().strip() == '':
                missing_fields.append(
                    "Computational budget" if algo == Algorithm.MULTI_FIDELITY.value else "Number of optimizations"
                )
            if algo == Algorithm.DOWNHILL.value and self.textbox_multiplier[
                "state"] == NORMAL and self.textbox_multiplier.get().strip() == '':
                missing_fields.append("Initial step")
            if self.theory is None: 
                missing_fields.append("Theoretical distribution (Histogram button)")

        # Display error
        if missing_fields:
            messagebox.showinfo("Warning", f"Fields not filled in:\n{', '.join(missing_fields)}")
            self.is_exit = True
            return

        # Parse values with error handling
        try:
            self.N = int(self.textbox_height.get().strip())
            self.mode = 1 if self.combobox_mode.get() == Mode.VAR.value else 0
            self.b = float(self.textbox_create.get().strip()) if self.textbox_create["state"] == NORMAL and self.textbox_create.get().strip() != '' else 0.0
            self.ts = float(self.textbox_ts.get().strip()) if self.textbox_ts["state"] == NORMAL and self.textbox_ts.get().strip() != '' else 0.0
            self.u = float(self.textbox_margin.get().strip()) if self.textbox_margin["state"] == NORMAL and self.textbox_margin.get().strip() != '' else 0.0
            self.current_G = int(self.textbox_count.get().strip()) if self.textbox_count["state"] == NORMAL and self.textbox_count.get().strip() != '' else 0

            if self.G == 0 or self.G is None:
                self.G = self.current_G

            opti_text = self.textbox_opti.get().strip() if self.textbox_opti["state"] == NORMAL else "1"
            try:
                opti_float = float(opti_text.replace(",", "."))
            except ValueError:
                opti_float = 1.0
            if algo == Algorithm.MULTI_FIDELITY.value:
                self.opti_budget = max(5.0, opti_float)
                self.opti_count = max(1, int(round(self.opti_budget)))
            else:
                self.opti_count = max(1, int(round(opti_float)))
                self.opti_budget = float(self.opti_count)

            mult_text = self.textbox_multiplier.get().strip() if self.textbox_multiplier["state"] == NORMAL else "0.1"
            self.multiplier = max(0.01, float(mult_text)) if mult_text.replace('.', '', 1).isdigit() else 0.1

            self.is_exit = False
        except ValueError as e:
            messagebox.showinfo("Warning", f"Invalid numeric values:\n{str(e)}")
            self.is_exit = True

    def configurate(self) -> None:
        self.window.title("Chemical Simulator")
        self.outer_pane.pack(fill=tk.BOTH, expand=True)
        self.outer_pane.add(self.sidebar, minsize=200, width=280)
        self.outer_pane.add(self.triple_pane, minsize=400)

        # Histogram, grid, text data
        self.triple_pane.add(self.graphbar, minsize=180)
        self.triple_pane.add(self.canvas_frame, minsize=200)
        self.triple_pane.add(self.statbar, minsize=200)

        self.btn_save_hist.pack(side=tk.LEFT, padx=4, pady=2)
        self.bar_top.pack(side=tk.TOP, fill=tk.X)
        self._graph_scrolled_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._bar_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._bar_pane.add(self.bar_host, minsize=90, stretch="always")
        self._bar_pane.add(self._bar_pane_filler, minsize=0, stretch="always")
        self.window.after(150, self._set_initial_bar_pane_sash)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self._sim_vsb.grid(row=0, column=1, sticky=tk.NS)
        self._sim_hsb.grid(row=1, column=0, sticky=tk.EW)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.bar_host.bind("<Configure>", self._on_bar_host_configure)
        self.canvas_frame.bind("<Configure>", self._on_canvas_frame_configure)
        self._stat_inner.bind("<Configure>", self._on_statbar_wrap_configure)
        self.canvas.bind("<MouseWheel>", self._scroll_sim_canvas)
        self.canvas.bind("<Button-4>", self._scroll_sim_canvas)
        self.canvas.bind("<Button-5>", self._scroll_sim_canvas)

        pwx = {"anchor": tk.W, "pady": 0}
        self.label_modelling.pack(anchor=tk.W, pady=(0, 1))
        self.combobox_algo.pack(fill=tk.X, pady=(0, 2))
        self.label_height.pack(**pwx)
        self.textbox_height.pack(fill=tk.X, pady=(0, 1))
        self.label_mode.pack(**pwx)
        self.combobox_mode.pack(fill=tk.X, pady=(0, 1))
        self.label_create.pack(**pwx)
        self.textbox_create.pack(fill=tk.X, pady=(0, 1))
        self.label_ts.pack(**pwx)
        self.textbox_ts.pack(fill=tk.X, pady=(0, 1))
        self.label_margin.pack(**pwx)
        self.textbox_margin.pack(fill=tk.X, pady=(0, 1))
        self.label_count.pack(**pwx)
        self.textbox_count.pack(fill=tk.X, pady=(0, 1))
        self.label_opti.pack(**pwx)
        self.textbox_opti.pack(fill=tk.X, pady=(0, 1))
        self.label_multiplier.pack(**pwx)
        self.textbox_multiplier.pack(fill=tk.X, pady=(0, 1))
        self.label_metric.pack(**pwx)
        self.combobox_metric.pack(fill=tk.X, pady=(0, 2))
        try:
            self.combobox_algo.configure(width=20)
            self.combobox_mode.configure(width=20)
        except tk.TclError:
            pass

        if not (self.combobox_algo.get() or "").strip():
            self.combobox_algo.set(Algorithm.SA.value)
        if not (self.combobox_mode.get() or "").strip():
            self.combobox_mode.set(Mode.CONST.value)
        try:
            from resources.simulator_config import apply_metric_from_config, load_config, metric_combobox_label

            _cfg = load_config()
            _m = apply_metric_from_config(_cfg)
            self.combobox_metric.set(metric_combobox_label(_m))
            from resources.simulator_config import apply_gui_defaults

            apply_gui_defaults(self, _cfg)
        except Exception as _e:
            if not (self.combobox_metric.get() or "").strip():
                self.combobox_metric.set(f"{METRIC_WEIGHTED_L} (old L)")
        self.change_opti_visible(None)
        self.change_label_create(None)

        # Run/Result, right: flags and Histogram (sb_actions)
        self._row_run.pack(fill=tk.X, pady=(0, 2))
        self._col_run_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._col_run_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.btb_run.pack(fill=tk.X, pady=1)
        self.btn_pause.pack(fill=tk.X, pady=1)
        self.btn_restart.pack(fill=tk.X, pady=1)
        self.btn_result.pack(fill=tk.X, pady=1)
        self.cb_run.pack(anchor=tk.W, fill=tk.X, pady=1)
        self.cb_bar.pack(anchor=tk.W, fill=tk.X, pady=1)
        self.btn_openhist.pack(fill=tk.X, pady=(2, 0))
        self._row_config.pack(fill=tk.X, pady=2)
        self.btn_openconfig.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_saveconfig.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_mode_compare.pack(fill=tk.X, pady=(4, 0))
        self._bar_scales.pack(fill=tk.X, pady=(2, 0))
        self.label_scale_bar.pack(anchor=tk.W, pady=(0, 0))
        self.scale_bar_mpl.pack(fill=tk.X, pady=0)
        self.label_scale_cv.pack(anchor=tk.W, pady=(2, 0))
        self.scale_cv_points.pack(fill=tk.X, pady=0)
        self.label_sleep.pack(anchor=tk.W, pady=(2, 0))
        self._label_sleep_value.pack(anchor=tk.W)
        self.scale_sleep.pack(fill=tk.X, pady=(0, 1))
        self._on_sleep_display(str(self._var_sleep.get()))
        # Bottom — actions, top — scrollable fields
        self.sb_actions.pack(side=tk.BOTTOM, fill=tk.X)
        self.sb_form_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for w in (self.stat_1, self.stat_2, self.stat_3, self.stat_compare):
            w.pack(anchor=tk.W, fill=tk.X, padx=4, pady=4)

        self.window.update_idletasks()
        try:
            h = max(int(self.outer_pane.winfo_height() or 1) // 2, 1)
            self.outer_pane.sash_place(0, 280, h)
        except tk.TclError:
            pass

        def _sb_panel_wheel(e) -> None:
            c = self._sb_scroll_canv
            if hasattr(e, "delta") and e.delta:
                c.yview_scroll(int(-1 * (e.delta / 120)), "units")
            else:
                n = int(getattr(e, "num", 0) or 0)
                if n == 4:
                    c.yview_scroll(-2, "units")
                elif n == 5:
                    c.yview_scroll(2, "units")

        self.sidebar.bind("<MouseWheel>", _sb_panel_wheel)
        for b in (4, 5):
            self.sidebar.bind(f"<Button-{b}>", _sb_panel_wheel)

    def run_btn(self) -> None:
        if self._method_status:
            messagebox.showwarning("Внимание", "Симуляция уже запущена. Дождитесь завершения или нажмите Пауза.")
            return

        self.set_params()
        if self.is_exit:
            return

        ```python
# Create the method only on the first run or after a restart
        if not self._method:
            algo = self.combobox_algo.get()

            if algo == Algorithm.DOWNHILL.value:
                self._method = CoordinateDescent(
                    rows=self.N,
                    addprob=self.b,
                    transitprob=self.ts,
                    mergeprob=self.u,
                    drawer=self,
                    sleeper=self,
                    multiplier=self.multiplier,
                    theory=self.theory
                )
            elif algo == Algorithm.TPE.value:
                self._method = OptunaIntegration(
                    rows=self.N,
                    theory=self.theory,  
                    drawer=self,
                    sleeper=self
                )
            elif algo == Algorithm.XNES.value:
                self._method = XnesOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.CMA_ES.value:
                self._method = CmaEsOptunaIntegration(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.GP_BAYES.value:
                self._method = GPBayesianOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.SKOPT_GP.value:
                self._method = SkoptGpOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.MULTI_FIDELITY.value:
                self._method = MultiFidelityOptimizer(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.PSO.value:
                self._method = PsoOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                    swarm_size=self._resolve_pso_swarm_size(self.opti_count),
                )
            elif algo == Algorithm.GA.value:
                self._method = GaOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.DE.value:
                self._method = DifferentialEvolutionOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.TRUST_REGION_BO.value:
                self._method = TrustRegionBoOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.TURBO_MT.value:
                self._method = TurboMtOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.HYPERBAND_TPE.value:
                self._method = OptunaHyperbandIntegration(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.LHS_NM.value:
                self._method = LhsNelderMeadOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.RF_EI.value:
                self._method = RfEiOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.CEM.value:
                self._method = CemOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            elif algo == Algorithm.SA.value:
                self._method = SimulatedAnnealingOptimization(
                    rows=self.N,
                    theory=self.theory,
                    drawer=self,
                    sleeper=self,
                )
            else:   
                self._method = DefaultRunner(
                    rows=self.N,
                    addprob=self.b,
                    transitprob=self.ts,
                    mergeprob=self.u,
                    drawer=self,
                    sleeper=self,
                    steps=self.G,
                    theory=self.theory
                )

            # Disable input fields
            self.textbox_height.config(state=DISABLED)
            self.combobox_mode.config(state=DISABLED)
            self.textbox_create.config(state=DISABLED)
            self.textbox_ts.config(state=DISABLED)
            self.textbox_margin.config(state=DISABLED)
            self.textbox_multiplier.config(state=DISABLED)
            self.textbox_opti.config(state=DISABLED)
            self.textbox_count.config(state=DISABLED)
            self.combobox_algo.config(state=DISABLED)
            self.btb_run.config(state=DISABLED)
            self.btn_pause.config(state=ACTIVE) 
            self.btn_restart.config(state=DISABLED)
            self.btn_result.config(state=NORMAL if hasattr(self._method, 'result') else DISABLED)
        else:
            # For a rerun, update the parameters
            algo = self.combobox_algo.get()
            if algo in ALGORITHMS_NEED_OPTI_OR_BUDGET:
                try:
                    opti_float = float((self.textbox_opti.get().strip() or "1").replace(",", "."))
                    if algo == Algorithm.MULTI_FIDELITY.value:
                        self.opti_budget = max(5.0, opti_float)
                        self.opti_count = max(1, int(round(self.opti_budget)))
                    else:
                        self.opti_count = max(1, int(round(opti_float)))
                        self.opti_budget = float(self.opti_count)
                except ValueError:
                    self.opti_count = 1
                    self.opti_budget = 1.0
            else:
                try:
                    self.current_G = max(1, int(self.textbox_count.get().strip() or "1"))
                    self._method.change_steps(self.current_G)
                except:
                    self.current_G = 1

        # Reset the stop flag before starting
        self.is_exit = False

        # RUN THE SIMULATION IN A SEPARATE THREAD (so as not to block the UI)
        import threading
        thread = threading.Thread(target=self._run_simulation, daemon=True)
        thread.start()

    def _finalize_simulation(self):
        """Final actions after the simulation completes"""
        print("\n Final rendering...")

        self.draw_stat()

        if not self.no_bar.get() and hasattr(self._method, 'create_bar'):
            try:
                bar = self._method.create_bar()
                if bar and len(bar) > 0:
                    print(f"Final bar: {len(bar)} values")
                    self.draw_bar(bar)
                else:
                    print(" Final bar is empty")
            except Exception as e:
                print(f" Error in final rendering: {e}")
                import traceback
                traceback.print_exc()

        self._show_canvas_after_run()

        # Автоматически активируем паузу/рестарт
        if not self.is_exit:
            self.pause_btn()

    def _run_simulation(self):
        """Actual simulation run in a separate thread"""
        algo = self.combobox_algo.get()

        if algo == Algorithm.TPE.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" TPE optimization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.HYPERBAND_TPE.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" Hyperband+TPE optimization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.XNES.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" xNES optimization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.CMA_ES.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" CMA-ES optimization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.GP_BAYES.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" GP (Bayesian optimization) error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.SKOPT_GP.value:
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" GP (skopt, EI) error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.MULTI_FIDELITY.value:
            self._method_status = True
            try:
                self._method.optimize(budget=self.opti_budget)
            except Exception as e:
                print(f" Multi-fidelity optimization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo in (
            Algorithm.PSO.value,
            Algorithm.GA.value,
            Algorithm.DE.value,
            Algorithm.TRUST_REGION_BO.value,
            Algorithm.TURBO_MT.value,
            Algorithm.LHS_NM.value,
            Algorithm.RF_EI.value,
            Algorithm.CEM.value,
            Algorithm.SA.value,
        ):
            self._method_status = True
            try:
                self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f" Optimization error ({algo}): {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        elif algo == Algorithm.DOWNHILL.value:
            # Coordinate descent
            self._method_status = True
            try:
                print(f" Starting Coordinate Descent: {self.opti_count} repetitions")

                for iteration in range(self.opti_count):
                    if self.is_exit:
                        break

                    print(f"\n=== Iteration {iteration + 1}/{self.opti_count} ===")
                    self._method.optimize()

                    # Update statistics and bar
                    if hasattr(self._method, '_hist') and len(self._method._hist) > 0:
                        last_result = self._method._hist[-1]
                        add, transit, merge, value, hist = last_result

                        # Update the UI
                        def update_ui(a=add, t=transit, m=merge, h=hist):
                            self.stat_1.config(text=f"Creation probability: {a:.5f}")
                            self.stat_2.config(text=f"Transition probability: {t:.5f}")
                            self.stat_3.config(text=f"Merge probability: {m:.5f}")

                            if h and not self.no_bar.get():
                                self.draw_bar(h)

                        self.window.after(0, update_ui)

                    if self.scale_sleep.get() > 0:
                        import time
                        time.sleep(self.scale_sleep.get())

                print("\n Coordinate Descent finished!")

            except Exception as e:
                print(f" Coordinate Descent error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._method_status = False
                self.window.after(0, self._finalize_simulation)

        else:
            # Normal run mode (DEFAULT)
            while self.current_G > 0 and not self.is_exit:
                self._method_status = True
                try:
                    self._method.run()
                except Exception as e:
                    print(f" Simulation error: {e}")
                    break
                finally:
                    self._method_status = False

                
                if not self.no_bar.get() and self.current_G % 10 == 0:
                    try:
                        bar = self._method.create_bar()
                        if bar:
                            self.window.after(0, lambda b=bar: self.draw_bar(b))
                    except Exception as e:
                        print(f" Bar rendering error: {e}")

                self.current_G -= 1

                if self.scale_sleep.get() > 0:
                    import time
                    time.sleep(self.scale_sleep.get())

            self.window.after(0, self._finalize_simulation)


    def sleep(self) -> None:
        # Batch mode comparison: the thread does not touch mainloop / after, otherwise Tk freezes and crashes
        if getattr(self, "_compare_modes_busy", False):
            return
        self.window.after(int(self.scale_sleep.get() * 1000))

    def prepare_draw(self) -> None:
        if getattr(self, "_compare_modes_busy", False):
            return
        if not self.no_ui.get() or self._is_drawing:
            self.canvas.delete("all")
            self._is_drawing &= False

    def draw_point(self, row: int, col: int, color_idx: int) -> None:
        if getattr(self, "_compare_modes_busy", False):
            return
        if not self.no_ui.get():
            self._is_drawing |= True
            k = max(0.2, self._canvas_area_scale.get())
            nrows = 0
            m = self._method
            if m is not None and hasattr(m, "rows"):
                nrows = int(getattr(m, "rows", 0) or 0)
            if nrows <= 0:
                nrows = int(self.N or 0)
            base = self.canvas.winfo_height() / max(1, nrows)
            diff = max(0.5, base * k)


            x1 = int(col * diff)
            y1 = int(row * diff)
            x2 = int(col * diff + diff)
            y2 = int(row * diff + diff)


            colors = ["#ffffff", "#e32636", "#00ffff", "#ffe135", "#1dacd6"]
            color = colors[0] if color_idx == 0 else colors[color_idx % len(colors)]

            try:
                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")
            except Exception as e:
                print(f"Error drawing point ({row},{col}): {e}")

    def complete_draw(self) -> None:
        if getattr(self, "_compare_modes_busy", False):
            return
        self.window.update()
        self._refresh_canvas_scrollregion()
        self.sleep()

    def _on_bar_mpl_scale(self, *args) -> None:
        self._apply_hist_fig_size_from_host()

    def _on_bar_pane_configure(self, event: Any) -> None:
        """PanedWindow divider movement — recompute the matplotlib size"""
        if event.widget is not self._bar_pane:
            return
        if self._bar_resize_job is not None:
            self.window.after_cancel(self._bar_resize_job)
        self._bar_resize_job = self.window.after(50, self._apply_hist_fig_size_from_host)

    def _set_initial_bar_pane_sash(self) -> None:
        """Initial divider position: more room for the histogram, below — an area for manual sizing"""
        try:
            if not self.window.winfo_exists():
                return
            self.window.update_idletasks()
            h = int(self._bar_pane.winfo_height())
            if h < 120:
                return
            y = max(100, min(int(h * 0.70), h - 48))
            self._bar_pane.sash_place(0, 0, y)
        except tk.TclError:
            pass

    def _on_bar_host_configure(self, event) -> None:
        if event.widget is not self.bar_host:
            return
        w = int(event.width)
        h = int(event.height)
        if w < 2 or h < 2:
            return
        if self._bar_resize_job is not None:
            self.window.after_cancel(self._bar_resize_job)
        self._bar_resize_job = self.window.after(120, self._apply_hist_fig_size_from_host)

    def _apply_hist_fig_size_from_host(self) -> None:
        self._bar_resize_job = None
        if not self._hist_figure or not self.graph:
            return
        try:
            if not self.window.winfo_exists():
                return
        except Exception:
            return
        self.window.update_idletasks()
        w = max(self.bar_host.winfo_width() - 6, 80)
        h = max(self.bar_host.winfo_height() - 4, 60)
        dpi = 88.0
        sc = max(0.2, self._bar_fig_scale.get())
        w_in = (w * sc) / dpi
        h_in = (h * sc) / dpi
        w_in, h_in = max(w_in, 0.3), max(h_in, 0.25)
        self._hist_figure.set_dpi(dpi)
        try:
            self._hist_figure.set_size_inches(w_in, h_in, forward=True)
        except Exception:
            return
        try:
            self._hist_figure.tight_layout(pad=0.4)
        except Exception:
            pass
        self.graph.draw_idle()

    def _on_canvas_frame_configure(self, event) -> None:
        if event.widget is not self.canvas_frame:
            return
        w = int(event.width)
        h = int(event.height)
        if w < 2 or h < 2:
            return
        if self._canvas_resize_job is not None:
            self.window.after_cancel(self._canvas_resize_job)
        self._canvas_resize_job = self.window.after(
            40, lambda: self._set_canvas_size_wh(w, h)
        )

    def _set_canvas_size_wh(self, w: int, h: int) -> None:
        self._canvas_resize_job = None
        try:
            self.canvas.config(width=max(w, 1), height=max(h, 1))
            self._refresh_canvas_scrollregion()
        except Exception:
            pass

    def _on_statbar_wrap_configure(self, event) -> None:
        if event.widget is not self._stat_inner:
            return
        w = int(event.width)
        if w > 24:
            self.stat_compare.config(wraplength=max(60, w - 16))

    def save_histogram(self) -> None:
        if not self._hist_figure:
            messagebox.showinfo("No histogram", "First wait for the bar to be drawn (the \"No bar\" button is unchecked).")
            return
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        low = path.lower()
        if any(low.endswith(suf) for suf in (".png", ".pdf", ".svg", ".eps")):
            ext = path.rsplit(".", 1)[-1].lower()
        else:
            ext = "png"
            path = f"{path}.png"
        try:
            self._hist_figure.savefig(
                path,
                format=ext,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                dpi=150,
            )
            messagebox.showinfo("Saved", path)
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _show_canvas_after_run(self) -> None:
        """
        Show the grid on the canvas after completion (all methods):
        - if the method has an up-to-date Board — draw it;
        - otherwise (inverse problem) — one run with the best/current probabilities, as in SimulationObjective.
        """
        if not self._method or self.no_ui.get():
            return
        try:
            if not self.window.winfo_exists():
                return
        except Exception:
            return

        rows = int(getattr(self._method, "rows", 0) or 0) or int(self.N or 0)
        if rows <= 0:
            return

        try:
            board = getattr(self._method, "board", None)
            if board is not None:
                self.prepare_draw()
                board.draw(self)
                self.window.update_idletasks()
                self.window.update()
                self._refresh_canvas_scrollregion()
                print(" Canvas: current Board drawn")
                return

            add = tr = mg = None
            if getattr(self._method, "best_params", None):
                p = self._method.best_params
                add = float(p.get("add", 0.0))
                tr = float(p.get("transit", 0.0))
                mg = float(p.get("merge", 0.0))
            elif hasattr(self._method, "probs") and self._method.probs and len(self._method.probs) >= 3:
                add = float(self._method.probs[0])
                tr = float(self._method.probs[1])
                mg = float(self._method.probs[2])
            else:
                add, tr, mg = float(self.b), float(self.ts), float(self.u)

            preview = Board(
                rows=rows,
                addprob=add,
                transitprob=tr,
                mergeprob=mg,
                current_mode=Mode.CONST,
            )
            for _ in range(STEPS_PER_TRIAL):
                preview.run()
            self.prepare_draw()
            preview.draw(self)
            self.window.update_idletasks()
            self.window.update()
            self._refresh_canvas_scrollregion()
            print("Canvas: preview run for the lattice")
        except Exception as e:
            print(f" Canvas rendering error: {e}")
            import traceback
            traceback.print_exc()

    def draw_bar(self, bar: dict) -> None:
        """Draw the cluster size distribution histogram"""
        try:
            if hasattr(self, 'no_bar') and self.no_bar.get():
                return
            if getattr(self, "_compare_modes_busy", False):
                return

            # Check that the bar is not empty
            if not bar or len(bar) == 0:
                print(" Bar is empty — nothing to draw")
                return

            # Check that the window exists
            if not hasattr(self, 'window') or self.window is None:
                return

            try:
                if not self.window.winfo_exists():
                    return
            except:
                return

            for ch in self.bar_host.winfo_children():
                try:
                    ch.destroy()
                except Exception:
                    pass
            self.graph = None
            self._hist_figure = None

            # Start with the base size; exact size — after packing (see _apply_hist_fig_size_from_host)
            fig = Figure(figsize=(3.2, 2.0), dpi=88)
            ax = fig.add_subplot(111)
            ax.set_xlabel("Cluster size (weight)", fontsize=8)
            ax.set_title("Theory / experiment", fontsize=9)

            sorted_keys = sorted(bar.keys())
            theory_keys = sorted(self.theory.keys()) if self.theory else []
            all_keys = sorted(set(sorted_keys) | set(theory_keys))

            x_index = {k: i for i, k in enumerate(all_keys)}

            # With theory: fractions p(k) for sizes ≤ max(theory); rare large k in one column
            if self.theory:
                s_t = float(sum(float(v) for v in self.theory.values()))
                s_e = float(sum(float(v) for v in bar.values()))
                if s_t <= 0 or s_e <= 0:
                    print(" Zero sum of theory or experiment — histogram not built")
                    return
                k_cap = max(float(x) for x in self.theory.keys())
                main_keys = sorted(
                    {k for k in (set(self.theory.keys()) | set(bar.keys())) if float(k) <= k_cap + 1e-9},
                    key=lambda z: float(z),
                )
                p_th = [float(self.theory.get(k, 0)) / s_t for k in main_keys]
                p_ex = [float(bar.get(k, 0)) / s_e for k in main_keys]
                p_ex_tail = sum(float(bar[k]) for k in bar if float(k) > k_cap) / s_e
                p_th_tail = sum(float(self.theory[k]) for k in self.theory if float(k) > k_cap) / s_t
                bar_labels: list = [str(k) for k in main_keys]
                if p_ex_tail > 1e-10 or p_th_tail > 1e-10:
                    p_th = p_th + [p_th_tail]
                    p_ex = p_ex + [p_ex_tail]
                    cap_lbl = f">{int(k_cap)}" if abs(k_cap - int(k_cap)) < 1e-9 else f">{k_cap:.1f}"
                    bar_labels = bar_labels + [cap_lbl]
                n = len(p_th)
                idx = list(range(n))
                ax.set_ylabel("Fraction p(k)", fontsize=8)
                w0 = 0.36
                ax.bar(
                    [i - w0 / 2 for i in idx],
                    p_th,
                    width=w0,
                    color="#1f77b4",
                    alpha=0.9,
                    edgecolor="#0d47a1",
                    hatch="//",
                    label="Theory",
                )
                ax.bar(
                    [i + w0 / 2 for i in idx],
                    p_ex,
                    width=w0,
                    color="#2ca02c",
                    alpha=0.65,
                    edgecolor="#1b5e20",
                    label="Experiment",
                )
                ymax = max(max(p_th) if p_th else 0, max(p_ex) if p_ex else 0, 0.08)
                ax.set_ylim(0, min(1.0, ymax * 1.25))
                x_positions = list(range(n))
                all_keys = bar_labels  # for consistency with set_xticklabels below
            else:
                x_positions = list(range(len(all_keys)))
                values = [bar[k] for k in sorted_keys]
                exp_x = [x_index[k] for k in sorted_keys]
                ax.set_ylabel("Number of clusters", fontsize=8)
                ax.bar(
                    exp_x,
                    values,
                    width=0.65,
                    color="#2ca02c",
                    alpha=0.65,
                    edgecolor="#1b5e20",
                    label="Experiment",
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(k) for k in all_keys], rotation=40, ha='right', fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.legend(loc="upper right", fontsize=7)

            if self.theory:
                compare_text = self._build_hist_compare_text(self.theory, bar)
                self.stat_compare.config(text=compare_text)
            else:
                self.stat_compare.config(text="")

            try:
                fig.tight_layout(pad=0.45)
            except Exception:
                pass

            self._hist_figure = fig
            self.graph = FigureCanvasTkAgg(fig, self.bar_host)
            self.graph.draw()
            self.graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.window.after(10, self._apply_hist_fig_size_from_host)

            print(f" Histogram drawn: {len(sorted_keys)} bars")

        except Exception as e:
            print(f" Histogram rendering error: {e}")
            import traceback
            traceback.print_exc()

    def _build_hist_compare_text(self, theory: dict, experimental: dict) -> str:
        """Metrics on fractions p(k) (normalized), as in the inverse problem. MAE/RMSE/MAPE
        are computed on p(k); L matches the optimizer (SA, PSO, GP)."""
        try:
            all_keys = sorted(set(theory.keys()) | set(experimental.keys()))
            if not all_keys:
                return "Comparison: not enough data"

            th_raw = [float(theory.get(k, 0.0)) for k in all_keys]
            ex_raw = [float(experimental.get(k, 0.0)) for k in all_keys]
            s_th = sum(th_raw)
            s_ex = sum(ex_raw)
            if s_th <= 0 or s_ex <= 0:
                return "Zero sum in theory or experiment"

            p_t = [t / s_th for t in th_raw]
            p_e = [e / s_ex for e in ex_raw]

            n = len(all_keys)
            mae = sum(abs(a - b) for a, b in zip(p_t, p_e)) / n
            rmse = (sum((a - b) ** 2 for a, b in zip(p_t, p_e)) / n) ** 0.5

            mape = 0.0
            nz = 0
            for a, b in zip(p_t, p_e):
                if a > 1e-12:
                    mape += abs((a - b) / a)
                    nz += 1
            mape = (mape / nz * 100.0) if nz else 0.0

            active_metric_name = get_active_metric()
            loss_fn = get_active_metric_function()
            l_opt = loss_fn(theory, experimental)

            msg = (
                f"L({active_metric_name})={l_opt:.4f} — the same metric as in SA/PSO/GP…\n"
                f"MAE(p)={mae:.4f} | RMSE(p)={rmse:.4f} | MAPE(p)={mape:.1f}%\n"
                f"Points={n} | Σtheor={s_th:.0f} Σexp={s_ex:.0f} (fractions p normalized to 1)"
            )
            print(msg.replace("\n", " | "))
            return msg
        except Exception as e:
            print(f" Error computing comparison metrics: {e}")
            return "Comparison: computation error"

    def draw_stat(self) -> None:
        """Display the current statistics"""
        try:
            if self._method and hasattr(self._method, 'probs') and self._method.probs:
                add = self._method.probs[0] if len(self._method.probs) > 0 else 0.0
                transit = self._method.probs[1] if len(self._method.probs) > 1 else 0.0
                merge = self._method.probs[2] if len(self._method.probs) > 2 else 0.0

                self.stat_1.config(text=f"Creation probability: {add:.5f}")
                self.stat_2.config(text=f"Transition probability: {transit:.5f}")
                self.stat_3.config(text=f"Merge probability: {merge:.5f}")
            elif self._method and hasattr(self._method, 'best_params') and self._method.best_params:
                params = self._method.best_params
                self.stat_1.config(text=f"Creation probability: {params.get('add', 0.0):.5f}")
                self.stat_2.config(text=f"Transition probability: {params.get('transit', 0.0):.5f}")
                merge_line = f"Merge probability: {params.get('merge', 0.0):.5f}"
                if hasattr(self._method, "budget_spent") and hasattr(self._method, "budget_total"):
                    merge_line += f" | budget: {self._method.budget_spent:.1f}/{self._method.budget_total:.1f}"
                self.stat_3.config(text=merge_line)
            else:
                self.stat_1.config(text=f"Creation probability: {self.b:.5f}")
                self.stat_2.config(text=f"Transition probability: {self.ts:.5f}")
                self.stat_3.config(text=f"Merge probability: {self.u:.5f}")
        except Exception as e:
            print(f"Error in draw_stat: {e}")
            self.stat_1.config(text="Statistics error")
            self.stat_2.config(text="")
            self.stat_3.config(text="")



    @run_time
    def run(self) -> None:
        algo = self.combobox_algo.get()

        if algo in ALGORITHMS_NEED_OPTI_OR_BUDGET:
            self._method_status = True
            try:
                if algo == Algorithm.MULTI_FIDELITY.value:
                    self._method.optimize(budget=self.opti_budget)
                else:
                    self._method.optimize(n_trials=self.opti_count)
            except Exception as e:
                print(f"Optimization error: {e}")
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.window.after(0,lambda msg=error_msg: messagebox.showerror("Error", f"Optimization interrupted:\n{msg}"))
            finally:
                self._method_status = False
                # Update the statistics in the main Tkinter thread
                self.window.after(0, self.draw_stat)
                self.window.after(0, self._gui_save_session_json)
        else:
            # Loop over simulation steps for the normal run mode 
            while self.current_G > 0 and not self.is_exit:
                self._method_status = True
                try:
                    self._method.run()
                except Exception as e:
                    print(f"Simulation error: {e}")
                    break
                finally:
                    self._method_status = False

                # Update the histogram in the main thread
                if not self.no_bar.get() and self.current_G % 10 == 0:  # Update less often for performance
                    try:
                        bar = self._method.create_bar()
                        self.window.after(0, lambda b=bar: self.draw_bar(b))
                    except:
                        pass

                self.current_G -= 1

                # Respect the user delay
                if self.scale_sleep.get() > 0:
                    import time
                    time.sleep(self.scale_sleep.get())

            # Final statistics update
            self.window.after(0, self.draw_stat)

        # Automatically activate pause after completion
        if not self.is_exit:
            self.window.after(0, self.pause_btn)

    def pause_btn(self) -> None:
        """Handle the pause button press — ONLY sets the stop flag"""
        self.is_exit = True  # Signal to the algorithms that a stop is needed

        # Defer the finalization of the interface state
        self.window.after(50, self._finalize_pause)

    def _finalize_pause(self):
        """Finalize the interface state AFTER the simulation stops"""
        # Update the button states
        self.btn_pause.config(state=DISABLED)
        self.btn_restart.config(state=ACTIVE)
        self.btb_run.config(state=ACTIVE)

        # Initialize the variable for the remaining trials/steps
        remaining = 0

        # Update the input fields depending on the mode
        algo = self.combobox_algo.get()

        if algo == Algorithm.DEFAULT.value:
            # Normal run mode (DEFAULT): update the remaining steps
            remaining = max(0, self.current_G)
            self.textbox_count.config(state=NORMAL)
            self.textbox_count.delete(0, 'end')
            self.textbox_count.insert(0, str(remaining))
        else:
            # Optimization algorithms: remaining trials or budget (MFO)
            self.textbox_opti.config(state=NORMAL)
            self.textbox_opti.delete(0, 'end')

            remaining = 0
            if algo == Algorithm.MULTI_FIDELITY.value and self._method and hasattr(self._method, "budget_total"):
                remaining = max(0.0, float(self._method.budget_total) - float(self._method.budget_spent))
                text = str(int(round(remaining))) if abs(remaining - round(remaining)) < 1e-6 else f"{remaining:.2f}"
                self.textbox_opti.insert(0, text)
            elif self._method and hasattr(self._method, "total_trials") and hasattr(self._method, "current_trial"):
                completed = max(0, self._method.current_trial - 1)
                remaining = max(0, self._method.total_trials - completed)
                self.textbox_opti.insert(0, str(remaining))
            else:
                self.textbox_opti.insert(0, "0")

        # Update the statistics
        self.draw_stat()

    def _gui_session_context(self) -> dict:
        ctx: dict = {"algorithm": self.combobox_algo.get(), "rows": getattr(self, "N", None)}
        if getattr(self, "theory", None):
            ctx["theory_preset"] = "loaded"
            ctx["theory_bins"] = len(self.theory)
        return ctx

    def _gui_save_session_json(self) -> None:
        if not self._method:
            return
        try:
            from resources.simulator_config import load_config, save_after_optimization

            cfg = load_config()
            if not cfg.get("save_session_json", True):
                return
            path = save_after_optimization(
                self._method,
                algo_label=self.combobox_algo.get(),
                context=self._gui_session_context(),
            )
            if path:
                print(f"Session JSON: {path}")
        except Exception as save_err:
            print(f"Failed to save simulator_session JSON: {save_err}")

    def result_btn(self) -> None:
        if self._method:
            print(f"Method type: {type(self._method)}")
            print(f"Has _hist attribute: {hasattr(self._method, '_hist')}")
            print(f"Has board attribute: {hasattr(self._method, 'board')}")

            try:
                result = self._method.result()
                print(f" Result saved: {result}")
                try:
                    from resources.simulator_config import build_session_payload, load_config, save_session_json

                    cfg = load_config()
                    if cfg.get("save_session_json", True):
                        payload = build_session_payload(
                            optimizer_name=type(self._method).__name__,
                            run_result=result,
                            extra=self._gui_session_context(),
                        )
                        session_path = save_session_json(payload, cfg)
                        print(f"Session JSON: {session_path}")
                except Exception as save_err:
                    print(f"Failed to save simulator_session JSON: {save_err}")
            except Exception as e:
                print(f" Error in result(): {e}")
                import traceback
                traceback.print_exc()

    def fields_clearing(self):
        self.textbox_height.delete(0, 'end')
        self.combobox_mode.delete(0, 'end')
        self.textbox_create.delete(0, 'end')
        self.textbox_ts.delete(0, 'end')
        self.textbox_margin.delete(0, 'end')
        self.textbox_count.delete(0, 'end')
        self.textbox_opti.delete(0, 'end')

    def restart_btn(self):
        if self._method:
            self._method = None
            self.G = 0

        self.theory = None

        self.textbox_height.config(state=NORMAL)
        self.combobox_mode.config(state=NORMAL)
        self.textbox_create.config(state=NORMAL)
        self.textbox_ts.config(state=NORMAL)
        self.textbox_margin.config(state=NORMAL)
        self.combobox_algo.config(state=NORMAL)
        self.textbox_count.config(state=NORMAL)

        self.btn_result.config(state=DISABLED)
        self.btn_restart.config(state=DISABLED)

        for ch in self.bar_host.winfo_children():
            try:
                ch.destroy()
            except Exception:
                pass
        self.graph = None
        self._hist_figure = None

        self.stat_1.config(text='')
        self.stat_2.config(text='')
        self.stat_3.config(text='')
        self.stat_compare.config(text='')

        self.canvas.delete("all")

    def change_label_create(self, _) -> None:
        if self.combobox_mode.get() == Mode.CONST.value:
            self.label_create.config(text="Creation probability")
        else:
            self.label_create.config(text="Right boundary of probability")

    def change_opti_visible(self, _) -> None:
        algo = self.combobox_algo.get()

        if algo == Algorithm.MULTI_FIDELITY.value:
            self.label_opti.config(text="Computational budget")
        else:
            self.label_opti.config(text="Number of optimizations")

        if algo not in ALGORITHMS_NEED_OPTI_OR_BUDGET:
            self.textbox_opti.delete(0, 'end')
            self.textbox_opti.config(state=DISABLED)
            self.textbox_count.config(state=NORMAL)
        else:
            self.textbox_opti.config(state=NORMAL)
            self.textbox_count.config(state=DISABLED)

        if algo != Algorithm.DOWNHILL.value:
            self.textbox_multiplier.delete(0, 'end')
            self.textbox_multiplier.config(state=DISABLED)
        else:
            self.textbox_multiplier.config(state=NORMAL)

        if algo in ALGORITHMS_INVERSE_PROBLEM:
            self.textbox_create.delete(0, 'end')
            self.textbox_create.config(state=DISABLED)
            self.textbox_ts.delete(0, 'end')
            self.textbox_ts.config(state=DISABLED)
            self.textbox_margin.delete(0, 'end')
            self.textbox_margin.config(state=DISABLED)
        else:
            self.textbox_create.config(state=NORMAL)
            self.textbox_ts.config(state=NORMAL)
            self.textbox_margin.config(state=NORMAL)

        if algo not in ALGORITHMS_USE_LOADED_HISTOGRAM:
            self.theory = None

        if algo == Algorithm.PSO.value:
            self.label_pso_swarm.pack(**pwx)
            self.textbox_pso_swarm.pack(fill=tk.X, pady=(0, 1))
        else:
            self.label_pso_swarm.pack_forget()
            self.textbox_pso_swarm.pack_forget()

    def _resolve_pso_swarm_size(self, n_trials: int) -> int:
        from classes.pso_optimization import resolve_swarm_size

        raw = "0"
        if hasattr(self, "textbox_pso_swarm"):
            raw = (self.textbox_pso_swarm.get() or "0").strip()
        try:
            requested = int(float(raw.replace(",", ".")))
        except (TypeError, ValueError):
            requested = 0
        return resolve_swarm_size(max(2, int(n_trials)), requested if requested > 0 else None)

    def update_optimization_stats(self, trial_num: int, total_trials: int, loss: float, params: dict):
        """
        Update optimization statistics in real time.
        Called from the optimization algorithms 
        """
        try:
            # Format the parameters
            add = params.get('add', params.get('addprob', 0.0))
            transit = params.get('transit', params.get('transitprob', 0.0))
            merge = params.get('merge', params.get('mergeprob', 0.0))

            # Update the labels
            self.stat_1.config(text=f"Trial: {trial_num}/{total_trials}")
            self.stat_2.config(text=f"Loss: {loss:.4f}")
            self.stat_3.config(text=f"add={add:.3f} | trans={transit:.3f} | merge={merge:.3f}")
            self.window.update_idletasks()

            if self.scale_sleep.get() > 0:
                self.window.after(int(self.scale_sleep.get() * 1000))

        except Exception as e:
            print(f"Statistics update error: {e}")

    def _instantiate_optimizer(self, algo: str):
        """New optimizer instance (does not assign self._method)"""
        th = self.theory
        if algo == Algorithm.DOWNHILL.value:
            return CoordinateDescent(
                rows=self.N,
                addprob=self.b,
                transitprob=self.ts,
                mergeprob=self.u,
                drawer=self,
                sleeper=self,
                multiplier=self.multiplier,
                theory=th,
            )
        if algo == Algorithm.TPE.value:
            return OptunaIntegration(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.XNES.value:
            return XnesOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.CMA_ES.value:
            return CmaEsOptunaIntegration(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.GP_BAYES.value:
            return GPBayesianOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.SKOPT_GP.value:
            return SkoptGpOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.MULTI_FIDELITY.value:
            return MultiFidelityOptimizer(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.PSO.value:
            return PsoOptimization(
                rows=self.N,
                theory=th,
                drawer=self,
                sleeper=self,
                swarm_size=self._resolve_pso_swarm_size(self.opti_count),
            )
        if algo == Algorithm.GA.value:
            return GaOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.DE.value:
            return DifferentialEvolutionOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.TRUST_REGION_BO.value:
            return TrustRegionBoOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.TURBO_MT.value:
            return TurboMtOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.HYPERBAND_TPE.value:
            return OptunaHyperbandIntegration(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.LHS_NM.value:
            return LhsNelderMeadOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.RF_EI.value:
            return RfEiOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.CEM.value:
            return CemOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.SA.value:
            return SimulatedAnnealingOptimization(rows=self.N, theory=th, drawer=self, sleeper=self)
        if algo == Algorithm.DEFAULT.value:
            return DefaultRunner(
                rows=self.N,
                addprob=self.b,
                transitprob=self.ts,
                mergeprob=self.u,
                drawer=self,
                sleeper=self,
                steps=self.G,
                theory=th,
            )
        return None

    @staticmethod
    def _loss_rank_key(item: tuple) -> tuple:
        """Sorting where finite L values are ascending, then +inf, then the rest"""
        try:
            bl = float(item[1])
        except (TypeError, ValueError):
            return (2, 0.0)
        if math.isnan(bl):
            return (2, 0.0)
        if math.isinf(bl) and bl > 0:
            return (1, 0.0)
        return (0, bl)

    def _coerce_mode_compare_loss(self, algo: str, m: Any, br: dict) -> float:
        if hasattr(m, "best_loss") and m.best_loss is not None:
            try:
                bl = float(m.best_loss)
                if math.isfinite(bl):
                    return bl
            except (TypeError, ValueError):
                pass
        th = self.theory
        if th and br:
            return float(get_active_metric_function()(th, br))
        return float("nan")

    def _open_mode_compare_dialog(self) -> None:
        if self._method_status:
            messagebox.showwarning("Comparison", "First wait for the current run to finish or pause")
            return
        if self._compare_modes_busy:
            messagebox.showinfo("Comparison", "A comparison is already running")
            return
        bp = benchmark_protocol_defaults()
        d = tk.Toplevel(self.window)
        d.title("Flexible method comparison")
        d.transient(self.window)
        d.grab_set()
        d.geometry("560x720")
        tk.Label(
            d,
            text=(
                "Select methods and protocol parameters. Result:\n"
                "• JSON in results/benchmark_flex-*.json\n"
                "• median L and p(k) plots in results/figures/"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=6)

        def _row(parent: tk.Misc, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
            fr = tk.Frame(parent)
            fr.pack(anchor=tk.W, padx=10, pady=2)
            tk.Label(fr, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
            tk.Spinbox(fr, from_=lo, to=hi, textvariable=var, width=8).pack(side=tk.LEFT)

        var_n_trials = tk.IntVar(value=int(bp.get("n_trials", 50)))
        var_n_repeats = tk.IntVar(value=int(bp.get("n_repeats", 3)))
        var_n_runs = tk.IntVar(value=int(bp.get("n_runs", 5)))
        _row(d, "n_trials (budget):", var_n_trials, 1, 500)
        _row(d, "n_repeats (stochasticity):", var_n_repeats, 1, 20)
        _row(d, "n_runs (independent):", var_n_runs, 1, 30)

        seed_fr = tk.LabelFrame(d, text="base_seed (LHS)")
        seed_fr.pack(anchor=tk.W, fill=tk.X, padx=10, pady=4)
        var_seed_mode = tk.StringVar(value="single")
        var_base_seed = tk.IntVar(value=int(bp.get("base_seed", 42)))
        seed_mode_fr = tk.Frame(seed_fr)
        seed_mode_fr.pack(anchor=tk.W, padx=6, pady=2)
        tk.Radiobutton(
            seed_mode_fr,
            text="Single seed",
            variable=var_seed_mode,
            value="single",
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            seed_mode_fr,
            text="Multiple seeds",
            variable=var_seed_mode,
            value="multi",
        ).pack(side=tk.LEFT, padx=(12, 0))

        single_seed_fr = tk.Frame(seed_fr)
        single_seed_fr.pack(anchor=tk.W, padx=6, pady=2)
        tk.Label(single_seed_fr, text="Seed:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Spinbox(
            single_seed_fr, from_=0, to=999_999, textvariable=var_base_seed, width=10
        ).pack(side=tk.LEFT)

        multi_seed_fr = tk.Frame(seed_fr)
        multi_seed_fr.pack(anchor=tk.W, padx=6, pady=2)
        seed_preset_vars: dict[int, tk.BooleanVar] = {}
        for s in SEED_PRESETS:
            seed_preset_vars[s] = tk.BooleanVar(value=s == 42)
            tk.Checkbutton(multi_seed_fr, text=str(s), variable=seed_preset_vars[s]).pack(
                side=tk.LEFT, padx=(0, 6)
            )
        extra_seed_fr = tk.Frame(seed_fr)
        extra_seed_fr.pack(anchor=tk.W, padx=6, pady=(0, 4))
        tk.Label(extra_seed_fr, text="Extra (comma-separated):", anchor=tk.W).pack(side=tk.LEFT)
        var_extra_seeds = tk.StringVar(value="")
        tk.Entry(extra_seed_fr, textvariable=var_extra_seeds, width=24).pack(side=tk.LEFT, padx=4)

        def _refresh_seed_panels(*_args: object) -> None:
            if var_seed_mode.get() == "single":
                single_seed_fr.pack(anchor=tk.W, padx=6, pady=2)
                multi_seed_fr.pack_forget()
                extra_seed_fr.pack_forget()
            else:
                single_seed_fr.pack_forget()
                multi_seed_fr.pack(anchor=tk.W, padx=6, pady=2)
                extra_seed_fr.pack(anchor=tk.W, padx=6, pady=(0, 4))

        var_seed_mode.trace_add("write", _refresh_seed_panels)
        _refresh_seed_panels()

        preset_fr = tk.Frame(d)
        preset_fr.pack(anchor=tk.W, padx=10, pady=4)
        tk.Label(preset_fr, text="Method preset:").pack(side=tk.LEFT)
        method_vars: dict[str, tk.BooleanVar] = {}
        method_param_vars: dict[str, dict[str, tk.Variable]] = {}
        method_param_frames: dict[str, tk.Frame] = {}
        for key, _label in list_selectable_benchmark_methods():
            method_vars[key] = tk.BooleanVar(value=key in BENCHMARK_PRESETS["series_q"])

        def _toggle_method_params(key: str, *_args: object) -> None:
            pfr = method_param_frames.get(key)
            if pfr is None:
                return
            if method_vars[key].get() and key in METHOD_PARAM_SCHEMA:
                pfr.pack(anchor=tk.W, padx=24, pady=(0, 4), fill=tk.X)
            else:
                pfr.pack_forget()

        def _apply_preset(name: str) -> None:
            keys = set(BENCHMARK_PRESETS.get(name, []))
            for k, v in method_vars.items():
                v.set(k in keys)
                _toggle_method_params(k)

        for pname, ptext in (
            ("series_q", "Series Q (4)"),
            ("pso_tpe", "PSO+TPE"),
            ("ch8_four", "Ch.8 (4)"),
            ("all", "All (17)"),
        ):
            tk.Button(
                preset_fr, text=ptext, width=10, command=lambda n=pname: _apply_preset(n)
            ).pack(side=tk.LEFT, padx=2)

        scroll_fr = tk.Frame(d)
        scroll_fr.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        canvas = tk.Canvas(scroll_fr, height=260)
        sb = tk.Scrollbar(scroll_fr, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for key, label in list_selectable_benchmark_methods():
            row_fr = tk.Frame(inner)
            row_fr.pack(anchor=tk.W, fill=tk.X, pady=1)
            cb = tk.Checkbutton(row_fr, text=f"{label}  [{key}]", variable=method_vars[key])
            cb.pack(anchor=tk.W)
            pfr = tk.Frame(row_fr)
            method_param_frames[key] = pfr
            specs = METHOD_PARAM_SCHEMA.get(key, [])
            if specs:
                method_param_vars[key] = {}
                defaults = default_method_options(key)
                if key == "pso" and int(bp.get("pso_swarm_size", 0)) > 0:
                    defaults["swarm_size"] = int(bp.get("pso_swarm_size", 0))
                for spec in specs:
                    p_row = tk.Frame(pfr)
                    p_row.pack(anchor=tk.W, pady=1)
                    tk.Label(p_row, text=spec["label"], width=24, anchor=tk.W).pack(side=tk.LEFT)
                    if spec["type"] == "int":
                        var_p = tk.IntVar(value=int(defaults.get(spec["key"], spec["default"])))
                        tk.Spinbox(
                            p_row,
                            from_=int(spec["min"]),
                            to=int(spec["max"]),
                            textvariable=var_p,
                            width=8,
                        ).pack(side=tk.LEFT)
                    else:
                        var_p = tk.DoubleVar(value=float(defaults.get(spec["key"], spec["default"])))
                        tk.Spinbox(
                            p_row,
                            from_=float(spec["min"]),
                            to=float(spec["max"]),
                            increment=0.01,
                            textvariable=var_p,
                            width=8,
                            format="%.3f",
                        ).pack(side=tk.LEFT)
                    method_param_vars[key][spec["key"]] = var_p
            method_vars[key].trace_add("write", lambda *_a, k=key: _toggle_method_params(k))
            _toggle_method_params(key)

        opt_fr = tk.Frame(d)
        opt_fr.pack(anchor=tk.W, padx=10, pady=4)
        var_show_fig = tk.BooleanVar(value=True)
        var_plot_median = tk.BooleanVar(value=True)
        var_plot_pk = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_fr, text="p(k) window after the run", variable=var_show_fig).pack(anchor=tk.W)
        tk.Checkbutton(opt_fr, text="Save median L plot (PNG)", variable=var_plot_median).pack(
            anchor=tk.W
        )
        tk.Checkbutton(opt_fr, text="Save top-5 p(k) plot (PNG)", variable=var_plot_pk).pack(
            anchor=tk.W
        )
        tk.Label(
            d,
            text="A loaded theory is required (Histogram). Time ≈ n_runs × methods × number of seeds.",
            justify=tk.LEFT,
            fg="#555",
        ).pack(anchor=tk.W, padx=10, pady=2)

        def _collect_method_options(selected: list[str]) -> dict[str, dict]:
            out: dict[str, dict] = {}
            for key in selected:
                if key not in method_param_vars:
                    continue
                raw = {pk: pv.get() for pk, pv in method_param_vars[key].items()}
                coerced = coerce_method_options(key, raw)
                if coerced:
                    out[key] = coerced
            return out

        def _collect_base_seeds() -> list[int]:
            if var_seed_mode.get() == "single":
                return [int(var_base_seed.get())]
            presets = [s for s, v in seed_preset_vars.items() if v.get()]
            return parse_seed_list(presets, var_extra_seeds.get(), fallback=int(var_base_seed.get()))

        def _on_go() -> None:
            selected = [k for k, v in method_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("Comparison", "Select at least one method.")
                return
            base_seeds = _collect_base_seeds()
            if not base_seeds:
                messagebox.showwarning("Comparison", "Specify at least one seed.")
                return
            d.destroy()
            self.set_params()
            if self.is_exit:
                return
            if self.theory is None or len(self.theory) < 1:
                messagebox.showwarning("Comparison", "First load the target histogram (Histogram).")
                return
            params = {
                "n_trials": int(var_n_trials.get()),
                "n_repeats": int(var_n_repeats.get()),
                "n_runs": int(var_n_runs.get()),
                "base_seeds": base_seeds,
                "method_options": _collect_method_options(selected),
                "show_figure": bool(var_show_fig.get()),
                "plot_median": bool(var_plot_median.get()),
                "plot_pk": bool(var_plot_pk.get()),
                "method_keys": selected,
                "param_low": float(bp.get("param_low", 0.0)),
                "param_high": float(bp.get("param_high", 1.0)),
            }
            self.btn_mode_compare.config(state=DISABLED)
            import threading

            def _w() -> None:
                self._thread_mode_compare(**params)

                def _on_done() -> None:
                    self.btn_mode_compare.config(state=NORMAL)

                self.window.after(0, _on_done)

            threading.Thread(target=_w, daemon=True).start()

        btn_fr = tk.Frame(d)
        btn_fr.pack(pady=8, padx=10, anchor=tk.W)
        tk.Button(btn_fr, text="Run", command=_on_go).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_fr, text="Cancel", command=d.destroy).pack(side=tk.LEFT)

    def _thread_mode_compare(
        self,
        *,
        n_trials: int,
        n_repeats: int,
        n_runs: int,
        base_seeds: Optional[list] = None,
        base_seed: int = 42,
        pso_swarm_size: int = 0,
        method_options: Optional[dict] = None,
        param_low: float,
        param_high: float,
        show_figure: bool = True,
        plot_median: bool = True,
        plot_pk: bool = True,
        method_keys: Optional[list] = None,
    ) -> None:
        self._compare_modes_busy = True
        results: list = []
        errors: list = []
        saved_path: Optional[str] = None
        saved_paths: List[str] = []
        figure_paths: List[str] = []
        keys = list(method_keys or BENCHMARK_PRESETS["series_q"])
        seeds = list(base_seeds) if base_seeds else [int(base_seed)]
        pso_opts = (method_options or {}).get("pso") or {}
        swarm_raw = pso_opts.get("swarm_size")
        swarm_eff = (
            int(swarm_raw)
            if swarm_raw is not None and int(swarm_raw) > 0
            else (int(pso_swarm_size) if int(pso_swarm_size) > 0 else None)
        )
        try:
            rows = int(self.N) if self.N else int(benchmark_protocol_defaults().get("rows", 12))
            theory = dict(self.theory or {})
            metric = get_active_metric()
            print(
                f"\n=== Comparison of {len(keys)} methods {keys} × {n_runs} runs × {len(seeds)} seed(s), "
                f"n_trials={n_trials}, n_repeats={n_repeats}, metric={metric} ===\n"
            )
            if method_options:
                print(f"  method_options: {method_options}")

            import datetime
            import json
            from pathlib import Path

            tag = "-".join(keys[:4])
            if len(keys) > 4:
                tag += f"-n{len(keys)}"

            for seed_idx, seed in enumerate(seeds):
                if self.is_exit:
                    break
                print(f"\n--- seed {seed} ({seed_idx + 1}/{len(seeds)}) ---\n")

                def _progress(step: int, _total: int, method_key: str) -> None:
                    print(f"  [{step}/{_total}] {method_key}")

                ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                seed_suffix = f"-seed{seed}" if len(seeds) > 1 else ""
                out_file = (
                    Path(__file__).resolve().parent.parent
                    / "results"
                    / f"benchmark_flex-{tag}{seed_suffix}-{ts}.json"
                )
                out_path = run_mode_compare_benchmark(
                    theory,
                    rows=rows,
                    n_trials=n_trials,
                    n_repeats=n_repeats,
                    n_runs=n_runs,
                    base_seed=int(seed),
                    param_low=param_low,
                    param_high=param_high,
                    metric=metric,
                    method_keys=keys,
                    pso_swarm_size=swarm_eff,
                    method_options=method_options,
                    base_seeds=seeds if len(seeds) > 1 else None,
                    output_path=out_file,
                    on_progress=_progress,
                    should_stop=lambda: self.is_exit,
                )
                saved_path = str(out_path)
                saved_paths.append(saved_path)
                print(f"\nBenchmark JSON saved: {saved_path}\n")

                if plot_median or plot_pk:
                    seed_figs = generate_benchmark_figures(
                        out_path,
                        plot_median=plot_median,
                        plot_pk_top=plot_pk,
                    )
                    figure_paths.extend(seed_figs)
                    for fp in seed_figs:
                        print(f"  Plot: {fp}")

                payload = json.loads(out_path.read_text(encoding="utf-8"))
                runs = (payload.get("targets") or [{}])[0].get("runs") or []
                if runs:
                    last_methods = runs[-1].get("methods") or {}
                    seed_results, seed_errors = mode_compare_results_from_run(
                        last_methods,
                        theory,
                        rows=rows,
                        n_repeats=n_repeats,
                        method_keys=keys,
                        sleeper=self,
                    )
                    if seed_idx == len(seeds) - 1:
                        results = seed_results
                        errors = seed_errors
                for row in (payload.get("summary") or {}).values():
                    print(f"--- Risk ranking (seed={seed}) ---")
                    for r in row.get("risk_ranking", []):
                        print(
                            f"  {int(r.get('risk_rank', 0)):2d}. {r.get('method', ''):<30} "
                            f"median={r.get('median_best_loss', float('nan')):.4f}"
                        )
                    print()

            if results:
                ranked = sorted(results, key=self._loss_rank_key)
                print("Ranking by L\n")
                for rank, (name, bl, _br) in enumerate(ranked, 1):
                    bls = f"{bl:.4f}" if math.isfinite(bl) else str(bl)
                    print(f"  {rank:2d}. {name[:55]:<55}  L={bls}")
                print()
        except Exception as e:  
            errors.append(str(e))
            print(f"Comparison: error: {e!s}")
        finally:
            self._compare_modes_busy = False

        r_copy = list(results)
        e_copy = list(errors)
        path_copy = saved_path
        paths_copy = list(saved_paths)
        fig_copy = list(figure_paths)

        def _finish() -> None:
            if path_copy or paths_copy:
                json_lines = "\n".join(paths_copy[:6]) if paths_copy else path_copy
                if len(paths_copy) > 6:
                    json_lines += f"\n… {len(paths_copy) - 6} more file(s)"
                extra = ""
                if fig_copy:
                    extra = "\n\nPlots:\n" + "\n".join(fig_copy[:4])
                messagebox.showinfo(
                    "Comparison finished",
                    f"JSON:\n{json_lines}{extra}\n\n"
                    "Compatible with plot_benchmark_results.py and plot_benchmark_hist_lines.py",
                )
            if show_figure and r_copy:
                self._show_mode_compare_figure(r_copy, e_copy)
            elif e_copy and not r_copy:
                messagebox.showwarning("Comparison", "\n".join(e_copy[:8]))

        self.window.after(0, _finish)

    def _populate_mode_compare_ax(
        self,
        ax,
        *,
        th: dict,
        all_ks: list,
        x_idx: list,
        p_theory: list,
        plottable: list,
        title: str,
        legend_outside: bool,
        thick_lines: bool,
        rank_offset: int = 0,
    ) -> List[Any]:
       """Draws the theory + algorithm curves; returns the legend (theory, then methods)."""
        ax.set_xlabel("Cluster size index", fontsize=9)
        ax.set_ylabel("Fraction p(k)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ms_theory = 7 if thick_lines else 5
        lw_theory = 2.4 if thick_lines else 1.5
        lines_out: List[Any] = []
        (ln_th,) = ax.plot(
            x_idx,
            p_theory,
            linestyle="-",
            marker="^",
            color="#111",
            label="Theory",
            markersize=ms_theory,
            linewidth=lw_theory,
            zorder=1000,
        )
        lines_out.append(ln_th)
        cmap = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        n_pt = len(plottable)
        ms = 5 if thick_lines else 3.5
        lw = 1.6 if thick_lines else 1.0
        base_alpha = 0.88 if thick_lines else 0.82
        for idx, (name, bl, bar) in enumerate(plottable):
            s_e = float(sum(float(x) for x in bar.values()))
            c = cmap[idx % len(cmap)]
            p_ex = [self._p_at_key(bar, kf, s_e) for kf in all_ks]
            if math.isfinite(bl):
                ls = f"  L={bl:.2f}"
            else:
                ls = f"  L={bl}"
            nm = name if len(name) < 36 else name[:33] + "…"
            rnk = rank_offset + idx + 1
            short = f"{rnk}. {nm}{ls}"
            z = 10 + (n_pt - idx)
            (ln,) = ax.plot(
                x_idx,
                p_ex,
                "o-",
                color=c,
                label=short,
                markersize=ms,
                linewidth=lw,
                alpha=base_alpha,
                zorder=z,
            )
            lines_out.append(ln)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([self._key_label(k) for k in all_ks], rotation=50, ha="right", fontsize=7)
        ymax = max(p_theory) if p_theory else 0.12
        for _n, _bl, bar in plottable:
            s_e = float(sum(float(x) for x in bar.values()))
            if s_e <= 0:
                continue
            row = [self._p_at_key(bar, kf, s_e) for kf in all_ks]
            if row:
                ymax = max(ymax, max(row))
        ax.set_ylim(0, min(1.0, max(ymax * 1.12, 0.12)))
        n_leg = 1 + n_pt
        leg = ax.legend(
            loc="center left" if legend_outside or n_leg > 6 else "upper left",
            bbox_to_anchor=(1.02, 0.5) if legend_outside or n_leg > 6 else (0.98, 0.98),
            fontsize=7 if thick_lines else 6,
            borderaxespad=0.8,
            framealpha=0.95,
        )
        if leg is not None:
            try:
                leg.set_draggable(False)
            except Exception:
                pass
        return lines_out

   @staticmethod
    def _wire_legend_hover_highlight(fig, ax, data_lines: List[Any]) -> None:
        """Highlight a curve when hovering over the corresponding legend entry """

        def dim_all(balanced: bool) -> None:
            for i, ln in enumerate(data_lines):
                if i == 0:
                    ln.set_alpha(1.0)
                    ln.set_linewidth(2.2)
                    if hasattr(ln, "set_markersize"):
                        ln.set_markersize(6)
                else:
                    if balanced:
                        ln.set_alpha(0.82)
                        ln.set_linewidth(1.05)
                        ln.set_markersize(3.8)
                    else:
                        ln.set_alpha(0.28)
                        ln.set_linewidth(0.85)
                        ln.set_markersize(3.0)

        def emphasize(idx: Optional[int]) -> None:
            if idx is None:
                dim_all(balanced=True)
                fig.canvas.draw_idle()
                return
            for i, ln in enumerate(data_lines):
                if i == idx:
                    ln.set_alpha(1.0)
                    ln.set_linewidth(2.8 if i > 0 else 2.6)
                    ln.set_zorder(2000 + i)
                    if hasattr(ln, "set_markersize"):
                        ln.set_markersize(8 if i > 0 else 7)
                else:
                    ln.set_alpha(0.22 if i > 0 else 0.45)
                    ln.set_linewidth(0.9)
                    if hasattr(ln, "set_markersize"):
                        ln.set_markersize(3.5)
                    ln.set_zorder(5 + i)
            fig.canvas.draw_idle()

        last_idx: List[Optional[int]] = [None]

        def on_motion(event) -> None:
            if event is None or event.x is None or event.y is None:
                return
            leg = ax.get_legend()
            if leg is None:
                return
            try:
                r = fig.canvas.get_renderer()
            except Exception:
                return
            inside_leg = False
            try:
                lf = leg.get_frame()
                if lf is not None:
                    bb_f = lf.get_window_extent(renderer=r)
                    inside_leg = bool(bb_f.contains(event.x, event.y))
            except Exception:
                pass
            idx_found: Optional[int] = None
            if inside_leg:
                for i, txt in enumerate(leg.get_texts()):
                    try:
                        bb = txt.get_window_extent(renderer=r)
                        try:
                            bb = bb.expanded(1.15, 1.35)
                        except Exception:
                            pass
                        if bb.contains(event.x, event.y):
                            idx_found = i
                            break
                    except Exception:
                        continue
            if idx_found != last_idx[0]:
                last_idx[0] = idx_found
                if idx_found is None:
                    dim_all(balanced=True)
                    for i, ln in enumerate(data_lines):
                        ln.set_zorder(10 + (len(data_lines) - i))
                    fig.canvas.draw_idle()
                else:
                    emphasize(idx_found)

        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        dim_all(balanced=True)

    def _show_mode_compare_figure(
        self,
        results: list,
        errors: list,
    ) -> None:
        th = self.theory
        if th is None or not results:
            if errors:
                messagebox.showerror(
                    "Comparison",
                    "No results were obtained.\n" + "\n".join(errors[:5]),
                )
            return
        s_t = float(sum(float(v) for v in th.values()))
        if s_t <= 0:
            return
        ranked = sorted(results, key=self._loss_rank_key)
        all_k_float = {float(x) for x in th.keys()}
        for _name, _bl, bar in results:
            all_k_float |= {float(x) for x in bar.keys()}
        all_ks = sorted(all_k_float)
        x_idx = list(range(len(all_ks)))
        p_theory = [self._p_at_key(th, kf, s_t) for kf in all_ks]
        plottable = [
            (name, bl, bar)
            for name, bl, bar in ranked
            if bar and float(sum(float(x) for x in bar.values())) > 0
        ]
        w = tk.Toplevel(self.window)
        w.title("Comparison: p(k) lines ")
        w.minsize(800, 480)
        try:
            w.state("zoomed")
        except tk.TclError:
            try:
                sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
                w.geometry(f"{sw}x{sh}+0+0")
            except tk.TclError:
                pass
        dpi = 100
        try:
            sw = w.winfo_screenwidth()
            sh = w.winfo_screenheight()
            fig_w = min(22.0, max(12.0, (sw - 120) / float(dpi)))
            fig_h = min(12.5, max(6.5, (sh - 220) / float(dpi)))
        except tk.TclError:
            fig_w, fig_h = 14.0, 8.0

        outer = tk.Frame(w)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        hint = tk.Label(
            outer,
            text="All methods"
            "Top-5",
            fg="#444",
            font=("TkDefaultFont", 8),
            wraplength=900,
            justify=tk.LEFT,
        )
        hint.pack(anchor=tk.W, padx=2, pady=(0, 4))
        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_all = tk.Frame(nb)
        tab_top = tk.Frame(nb)
        nb.add(tab_all, text="All methods")
        nb.add(tab_top, text="Top-5")

        err_note = f"Errors: {len(errors)}; see console" if errors else ""

        # all methods 
        fig1 = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax1 = fig1.add_subplot(1, 1, 1)
        lines1 = self._populate_mode_compare_ax(
            ax1,
            th=th,
            all_ks=all_ks,
            x_idx=x_idx,
            p_theory=p_theory,
            plottable=plottable,
            title="Theory and best p(k) by algorithm; the legend shows the rank by L",
            legend_outside=True,
            thick_lines=False,
            rank_offset=0,
        )
        if err_note:
            fig1.text(0.5, 0.015, err_note, ha="center", fontsize=8, color="#a00")
        try:
            fig1.tight_layout(rect=[0, 0.04, 0.78, 0.96])
        except Exception:
            pass
        cnv1 = FigureCanvasTkAgg(fig1, tab_all)
        cnv1.draw()
        cnv1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._wire_legend_hover_highlight(fig1, ax1, lines1)

        #  top-5 
        top5 = plottable[:5]
        fig2 = Figure(figsize=(fig_w, fig_h * 0.92), dpi=dpi)
        ax2 = fig2.add_subplot(1, 1, 1)
        lines2 = self._populate_mode_compare_ax(
            ax2,
            th=th,
            all_ks=all_ks,
            x_idx=x_idx,
            p_theory=p_theory,
            plottable=top5,
            title="Five best methods by L",
            legend_outside=len(top5) > 3,
            thick_lines=True,
            rank_offset=0,
        )
        if err_note:
            fig2.text(0.5, 0.015, err_note, ha="center", fontsize=8, color="#a00")
        try:
            fig2.tight_layout(rect=[0, 0.04, 0.82, 0.96])
        except Exception:
            pass
        cnv2 = FigureCanvasTkAgg(fig2, tab_top)
        cnv2.draw()
        cnv2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        if len(top5) > 0:
            self._wire_legend_hover_highlight(fig2, ax2, lines2)

        if errors:
            ttk.Separator(w, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)
            err_fr = tk.Frame(w)
            err_fr.pack(fill=tk.BOTH, padx=6, pady=4)
            tk.Label(err_fr, text="Errors (excerpt):", font=("TkDefaultFont", 8, "bold")).pack(anchor=tk.W)
            tk.Text(err_fr, height=4, wrap=tk.WORD, font=("Consolas", 8), bg="#fff5f5").pack(
                fill=tk.BOTH, expand=True
            )
            tx = err_fr.winfo_children()[-1]  # type: ignore[assignment]
            tx.insert("1.0", "\n".join(errors))

    @staticmethod
    def _p_at_key(bar: dict, kf: float, s: float) -> float:
        for key in bar:
            if abs(float(key) - kf) < 1e-9:
                return float(bar[key]) / s
        return 0.0

    @staticmethod
    def _key_label(kf: float) -> str:
        if abs(kf - int(kf)) < 1e-9:
            return str(int(round(kf)))
        return f"{kf:g}"

    def start(self) -> None:
        self.configurate()
        self.window.mainloop()

    def _update_downhill_stats(self, result_tuple):
        """Update statistics for coordinate descent"""
        try:
            # result_tuple = (add, transit, merge, value, hist) — iteration result tuple
            add, transit, merge, value, hist = result_tuple
            self.stat_1.config(text=f"Creation probability: {add:.5f}")
            self.stat_2.config(text=f"Transition probability: {transit:.5f}")
            self.stat_3.config(text=f"Merge probability: {merge:.5f}")
            self.window.update_idletasks()
        except Exception as e:
            print(f"DOWNHILL statistics update error: {e}")