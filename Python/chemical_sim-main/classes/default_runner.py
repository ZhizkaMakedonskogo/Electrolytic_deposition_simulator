import datetime

from classes import Board
from classes.base_comparator import BaseComparator
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.utilities import optimization_result_file_path


class DefaultRunner(BaseComparator):
    def __init__(
            self,
            rows: int,
            addprob: float,
            transitprob: float,
            mergeprob: float,
            theory: dict,
            drawer: Drawer,
            sleeper: Sleeper,
            steps: int
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self.board = Board(rows=rows, addprob=addprob, transitprob=transitprob, mergeprob=mergeprob)
        self.steps = steps
        self.current_steps = 0

    def run(self):
        if self.theory is None:
            # Simple simulation with a fixed number of steps
            if self.current_steps == 0:
                self.current_steps = self.steps
            if self.current_steps > 0:
                self._modelling()
                self.current_steps -= 1
        else:
            # Simulation with theory
            self.modelling()

    def modelling(self):
        if self.theory is None:
            self.current_steps = self.steps
            while self.current_steps > 0:
                self._modelling()
                self.current_steps -= 1
        else:
            super().modelling()

    def optimize(self):
        self.modelling()

    def change_steps(self, new_steps: int):
        self.steps = new_steps

    def _write_board_analysis_report(self, file_path: str) -> None:
        weight_histogram = self.board.create_bar()
        clusters_info = self.board.clusters_stats()
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('WeightAnalysis:\n{')
            for key, value in weight_histogram.items():
                file.write(f'{key}: {value}, ')
            file.write('}\n\nClusterAnalysis:\n' + clusters_info)

    def result(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        file_path = optimization_result_file_path("BoardAnalysis", timestamp)
        self._write_board_analysis_report(file_path)