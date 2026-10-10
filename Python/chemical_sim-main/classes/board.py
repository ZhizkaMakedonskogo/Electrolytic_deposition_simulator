
from . import Cell
from . import Cluster

import random
import statistics

from interfaces.drawer import Drawer
from interfaces.drawable import Drawable

from resources.constants import RESIZE, Status, Mode

random.seed()


class Board(Drawable):
    def __init__(
            self,
            rows: int,
            addprob: float,
            transitprob: float,
            mergeprob: float,
            current_mode: Mode = Mode.CONST
    ):
        self._rows = rows
        self._columns = RESIZE
        self._place = [[-1] * self._columns for _ in range(self._rows)]
        self._clusters = {}
        self._current_mode = current_mode
        self._create_atom = addprob
        self._transit_atom = transitprob
        self._merge_cluster = mergeprob
        self._atoms_count = 0

    @property
    def rows(self) -> int:
        return self._rows

    def _resize(self) -> None:
        """Expands the grid in one dimension by the RESIZE constant"""

        for row in self._place:
            row += [-1] * RESIZE
        self._columns += RESIZE

    def _add_atom(self, row: int) -> Cell:
        """Adds an atom to the given row. If the row is full, expands the grid"""

        j = self._place[row].index(-1) if -1 in self._place[row] else self._columns
        if j >= self._columns:
            self._resize()
        self._place[row][j] = 0

        return Cell(row, j)

    def _atom_transition(self, current: Cell) -> Cell:
        """Moves an atom to an adjacent cell on the surface. Chooses two paths: up or down along
        the surface. It can move it to a free cell or on top of another atom. If both directions
        are blocked, returns the input parameter — i.e., does not move."""

        is_up = random.choice((True, False))

        self._place[current.x][current.y] = -1

        if is_up and current.x - 1 >= 0:
            current = self._add_atom(current.x - 1)
        elif not is_up and current.x + 1 < self._rows:
            current = self._add_atom(current.x + 1)

        return current

    def _check_cluster(self, current: Cell, candidates: list, temp_place: list) -> None:
        """Recursive function that searches for candidates to add to existing clusters or to create a new one"""

        x = current.x
        y = current.y

        if self._place[x][y] == -1:
            temp_place[x][y] = 1
            return
        elif temp_place[x][y] == 1:
            return

        temp_place[x][y] = 1

        cur_val = self._place[x][y]

        if self._is_interior_with_all_neighbors_equal(x, y, cur_val):
            return

        candidates.append(Cell(x, y))

        if x - 1 >= 0:
            self._check_cluster(Cell(x - 1, y), candidates, temp_place)

        if x + 1 < self._rows:
            self._check_cluster(Cell(x + 1, y), candidates, temp_place)

        if y - 1 >= 0:
            self._check_cluster(Cell(x, y - 1), candidates, temp_place)

        if y + 1 < self._columns:
            self._check_cluster(Cell(x, y + 1), candidates, temp_place)

        if x - 1 >= 0 and y - 1 >= 0:
            self._check_cluster(Cell(x - 1, y - 1), candidates, temp_place)

        if x + 1 < self._rows and y - 1 >= 0:
            self._check_cluster(Cell(x + 1, y - 1), candidates, temp_place)

        if x - 1 >= 0 and y + 1 < self._columns:
            self._check_cluster(Cell(x - 1, y + 1), candidates, temp_place)

        if x + 1 < self._rows and y + 1 < self._columns:
            self._check_cluster(Cell(x + 1, y + 1), candidates, temp_place)

    def _is_interior_with_all_neighbors_equal(self, x: int, y: int, cur_val: int) -> bool:
        """Interior 3×3 cell with the same value in all 8 neighboring cells (early DFS exit)"""

        if not (x - 1 >= 0 and x + 1 < self._rows and y - 1 >= 0 and y + 1 < self._columns):
            return False
        p = self._place
        return (
            p[x - 1][y] == cur_val
            and p[x + 1][y] == cur_val
            and p[x][y - 1] == cur_val
            and p[x][y + 1] == cur_val
            and p[x + 1][y + 1] == cur_val
            and p[x - 1][y + 1] == cur_val
            and p[x + 1][y - 1] == cur_val
            and p[x - 1][y - 1] == cur_val
        )

    def _neighbor_cluster_id_for_free_atom(self, atom: Cell) -> int:
        """Cluster ID of a neighbor; when there are several neighbors, the last check wins"""

        new_found = -1
        x, y = atom.x, atom.y
        p = self._place
        r, c = self._rows, self._columns

        def consider(nx: int, ny: int) -> None:
            nonlocal new_found
            if p[nx][ny] > 0:
                new_found = p[nx][ny]

        if x - 1 >= 0:
            consider(x - 1, y)
        if x + 1 < r:
            consider(x + 1, y)
        if y - 1 >= 0:
            consider(x, y - 1)
        if y + 1 < c:
            consider(x, y + 1)
        if x - 1 >= 0 and y - 1 >= 0:
            consider(x - 1, y - 1)
        if x + 1 < r and y - 1 >= 0:
            consider(x + 1, y - 1)
        if x - 1 >= 0 and y + 1 < c:
            consider(x - 1, y + 1)
        if x + 1 < r and y + 1 < c:
            consider(x + 1, y + 1)
        return new_found

    def _create_cluster(self, current: Cell) -> None:
        """Creates a cluster if there are 4 free atoms in the neighborhood, or adds free atoms to
        existing clusters, or merges clusters"""

        temp_place = [[0] * self._columns for _ in range(self._rows)]
        candidates = []
        self._check_cluster(current, candidates, temp_place)

        min_clusternumber = 0
        for atom in candidates:
            if self._place[atom.x][atom.y] != 0:
                if min_clusternumber == 0:
                    min_clusternumber = self._place[atom.x][atom.y]
                else:
                    min_clusternumber = min(min_clusternumber, self._place[atom.x][atom.y])

        if min_clusternumber == 0:
            if len(candidates) >= 4:
                new_cluster = Cluster(limiter=self._rows)
                for atom in candidates:
                    new_cluster.add_atom(atom)
                self._clusters[new_cluster.number] = new_cluster
                self._cluster_coloring(self._clusters.get(new_cluster.number))
        else:
            merging = set()
            for atom in candidates:
                if self._place[atom.x][atom.y] == min_clusternumber:
                    continue
                elif self._place[atom.x][atom.y] != 0:
                    merging.add(self._place[atom.x][atom.y])
                else:
                    new_found = self._neighbor_cluster_id_for_free_atom(atom)

                    if new_found != -1:
                        self._clusters[new_found].add_atom(atom)
                        self._cluster_coloring(self._clusters.get(new_found))
            if len(merging) > 0:
                self._cluster_merger([i for i in merging] + [min_clusternumber])  

    def _has_adjacent_occupied_cell_excluding_cluster(self, x: int, y: int, cluster_number: int) -> bool:
        """Adjacent occupied cell different from the current cluster number (-1 is considered free)"""

        p = self._place
        r, c = self._rows, self._columns

        def foreign(nx: int, ny: int) -> bool:
            return p[nx][ny] != cluster_number and p[nx][ny] != -1

        return (
            x - 1 >= 0 and foreign(x - 1, y)
            or x + 1 < r and foreign(x + 1, y)
            or y - 1 >= 0 and foreign(x, y - 1)
            or y + 1 < c and foreign(x, y + 1)
            or x - 1 >= 0 and y - 1 >= 0 and foreign(x - 1, y - 1)
            or x - 1 >= 0 and y + 1 < c and foreign(x - 1, y + 1)
            or x + 1 < r and y - 1 >= 0 and foreign(x + 1, y - 1)
            or x + 1 < r and y + 1 < c and foreign(x + 1, y + 1)
        )

    def _check_cluster_for_clusters(self, found: int) -> None:
        """Determines whether there are other clusters or free atoms in the neighborhood of the given cluster.
        If there are, it starts the create_cluster procedure"""

        cluster_number = self._clusters[found].number
        for i in range(self._clusters[found].size()):
            x = self._clusters[found].atoms[i].x
            y = self._clusters[found].atoms[i].y
            if 0 <= x < self._rows and 0 <= y < self._columns:
                if self._has_adjacent_occupied_cell_excluding_cluster(x, y, cluster_number):
                    self._create_cluster(self._clusters[found].atoms[i])
                    return

    def _cluster_uncoloring(self, cluster: Cluster) -> None:
        """Removes the cluster's atoms from the grid"""

        for item in cluster.atoms:
            if 0 <= item.x < self._rows and 0 <= item.y < self._columns:
                self._place[item.x][item.y] = -1

        cluster.is_colored = False

    def _cluster_coloring(self, cluster: Cluster) -> None:
        """Draws the cluster's atoms on the grid"""

        for item in cluster.atoms:
            if item.y >= self._columns:
                self._resize()
            if 0 <= item.x < self._rows and 0 <= item.y < self._columns:
                self._place[item.x][item.y] = cluster.number

        cluster.is_colored = True

    def _cluster_merger(self, candidates: list) -> None:
        """Cluster merging procedure. If there are more than 2 clusters, then all clusters merge into one; otherwise,
        based on a random number generator, it either merges the clusters into one or pushes them apart,
        depending on the cluster sizes"""

        if len(candidates) > 2:
            for i in range(len(candidates) - 1):
                self._clusters[candidates[-1]].merger(self._clusters[candidates[i]])
                self._clusters[candidates[i]].status = Status.MERGING
                self._cluster_uncoloring(self._clusters[candidates[i]])
            self._cluster_coloring(self._clusters.get(candidates[-1]))
        else:
            randmerge = random.random()
            if randmerge <= self._merge_cluster or self._clusters[candidates[0]].status == Status.ON_SURFACE or \
                    self._clusters[candidates[1]].status == Status.ON_SURFACE:
                self._clusters[candidates[1]].merger(self._clusters[candidates[0]])
                self._cluster_uncoloring(self._clusters[candidates[0]])
                self._cluster_coloring(self._clusters[candidates[1]])
                self._clusters[candidates[0]].status = Status.MERGING
            else:
                if self._clusters[candidates[0]].size() >= self._clusters[candidates[1]].size():
                    repulsion_index = candidates[1]
                    stay_index = candidates[0]
                else:
                    repulsion_index = candidates[0]
                    stay_index = candidates[1]

                if self._clusters[repulsion_index].border_left().x >= self._clusters[stay_index].border_right().x:
                    if self._clusters[repulsion_index].status.UP_ALONG_SURFACE:
                        self._clusters[repulsion_index].status = Status.DOWN_ALONG_SURFACE
                else:
                    if self._clusters[repulsion_index].status != Status.UP_ALONG_SURFACE:
                        self._clusters[repulsion_index].status = Status.UP_ALONG_SURFACE

                self._cluster_uncoloring(self._clusters.get(repulsion_index))
                self._clusters[repulsion_index].transition()
                self._cluster_coloring(self._clusters.get(repulsion_index))

    def _queue_transit(self) -> None:
        """Cluster motion procedure (discrete-model imitation): lifts clusters with LEAVING status
        off the surface, moves clusters with UP_ALONG_SURFACE, DOWN_ALONG_SURFACE statuses along the surface"""

        for cluster in self._clusters.values():
            if cluster.status == Status.OFF_SURFACE and cluster.is_colored:
                cluster.is_colored = not cluster.is_colored
                self._cluster_uncoloring(cluster)
            if cluster.status in (Status.MERGING, Status.ON_SURFACE, Status.OFF_SURFACE):
                continue

            if cluster.status == Status.DOWN_ALONG_SURFACE or cluster.status == Status.UP_ALONG_SURFACE:
                speed = cluster.speed()
                if speed >= self._rows:
                    speed -= 1
                _exit = False

                j = speed
                while j > 0 and not _exit:
                    self._check_cluster_for_clusters(cluster.number)
                    if cluster.status in (Status.MERGING, Status.ON_SURFACE, Status.OFF_SURFACE):
                        _exit = True
                        continue
                    self._cluster_uncoloring(self._clusters.get(cluster.number))
                    cluster.transition()
                    self._cluster_coloring(self._clusters.get(cluster.number))
                    j -= 1

            if cluster.status == Status.LEAVING:
                if cluster.can_leaving():
                    self._check_cluster_for_clusters(cluster.number)
                    if cluster.status in (Status.MERGING, Status.ON_SURFACE, Status.OFF_SURFACE):
                        continue
                    self._cluster_uncoloring(self._clusters.get(cluster.number))
                    cluster.leave()
                    self._cluster_coloring(self._clusters.get(cluster.number))
                elif cluster.border_left().y != 0:
                    cluster.status = random.choice((Status.UP_ALONG_SURFACE, Status.DOWN_ALONG_SURFACE))

    def run(self) -> None:
        """Method that performs one modeling step"""

        create_atom_prob = self._create_atom
        if self._current_mode == Mode.VAR:
            create_atom_prob = random.random()

        rand_create = random.random()
        if rand_create <= create_atom_prob:
            self._atoms_count += 1
            position = random.randint(0, self._rows - 1)
            current = self._add_atom(position)
            self._create_cluster(current)
            if self._place[current.x][current.y] == 0:
                rand_transition = random.random()
                if rand_transition <= self._transit_atom:
                    current = self._atom_transition(current)                                                     
                    self._create_cluster(current)                                                               
            if self._place[current.x][current.y] > 0:
                rand_leaving = random.random()
                cluster_number = self._place[current.x][current.y]
                leaving_prob = 1.0 - self._clusters[cluster_number].adjoined / self._clusters[cluster_number].size()
                if rand_leaving < leaving_prob:
                    self._clusters.get(self._place[current.x][current.y]).status = Status.LEAVING
        self._queue_transit()  # TODO: add a check for whether it needs to run; as an option, add counters per cluster type and run queue only when needed

    def draw(self, dr: Drawer) -> None:
        """Method that draws the current grid using a Drawer object"""

        for row in range(self._rows):
            for col in range(self._columns):
                if self._place[row][col] != -1:
                    dr.draw_point(row, col, self._place[row][col])

    def create_bar(self) -> dict:
        """Creates a histogram of the obtained modeling results"""

        bar: dict[int, int] = {}
        for cluster in self._clusters.values():
            size = cluster.size()
            bar[size] = bar.get(size, 0) + 1
        return bar

    def stats(self) -> dict:
        """Returns statistics of the completed modeling run"""

        sizes = [cluster.size() for cluster in self._clusters.values()] if len(self._clusters) > 0 else [0]

        avg = statistics.mean(sizes)
        med = statistics.median(sizes)
        span = sizes[-1] - sizes[0]

        return {
            'atoms': self._atoms_count,
            'avg': avg,
            'med': med,
            'span': span,
            'clusters_count': len(self._clusters)
        }  # TODO: make a validating class?

    def clusters_stats(self) -> str:
        """Creates a file with a description for each cluster obtained as a result of modeling"""

        clusters_info = ''

        for cluster in self._clusters.values():
            if cluster.status == Status.MERGING:
                continue

            clusters_info += f'Cluster number: {cluster.number}\n' \
                             f'Cluster size: {cluster.size()}\n' \
                             f'Cluster bounds: ' \
                             f'vertically: {cluster.border_right().x}, {cluster.border_left().x}; ' \
                             f'horizontally: {cluster.border_right().y}, {cluster.border_left().y}\n' \
                             f'Status: {cluster.status.value}\n\n' \
                             f'Cluster image:\n\n'
            clusters_info += cluster.image() + '\n'

        return clusters_info