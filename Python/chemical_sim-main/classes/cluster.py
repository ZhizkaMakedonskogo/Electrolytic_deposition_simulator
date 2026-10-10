from . import Cell
from resources.constants import Status


class Cluster:
    IndexCluster = 0

    def __init__(self, limiter: int):
        self._atoms = []
        self._status = Status.ON_SURFACE
        self._number = Cluster.IndexCluster + 1
        Cluster.IndexCluster += 1
        self._adjoined = 0
        self.min = Cell()  
        self.max = Cell()
        self.leave_limit = 0
        self.limiter = limiter
        self.is_colored = False  

    @property
    def atoms(self) -> list:
        return list(self._atoms) 

    @property
    def number(self) -> int:
        return self._number

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, new_status: Status):
        self._status = new_status

    @property
    def adjoined(self) -> int:
        return self._adjoined

    def size(self) -> int:
        return len(self._atoms)

    def speed(self) -> int:
        return self.size()  
    def border_right(self) -> Cell:
        return self.max

    def border_left(self) -> Cell:
        return self.min

    def can_leaving(self) -> bool:
        if self.border_right().y >= self.leave_limit:
            return False
        return True

    def _define_rect_border(self) -> None:
        self._adjoined = sum([1 if i.y == 0 else 0 for i in self._atoms])
        x = [i.x for i in self._atoms]
        y = [i.y for i in self._atoms]
        self.max.x = max(x)
        self.max.y = max(y)
        self.min.x = min(x)
        self.min.y = min(y)

        if self.border_left().y == 0 and self.status != Status.LEAVING:
            self._status = Status.ON_SURFACE

        if self.border_right().x < 0 or self.border_left().x >= self.limiter:
            self._status = Status.OFF_SURFACE

        self.leave_limit = (self.size() / 2) if self.size() / 2 < self.limiter else self.limiter - 1

    def add_atom(self, atom: Cell) -> None:
        self._atoms.append(atom)

        self.max.x = max(atom.x, self.max.x)
        self.max.y = max(atom.y, self.max.y)
        self.min.x = min(atom.x, self.min.x)
        self.min.y = min(atom.y, self.min.y)

    def leave(self) -> None:
        for j in range(self.size()):
            self._atoms[j].y += 1
        self._status = Status.LEAVING
        self._define_rect_border()

    def transition(self) -> None:
        for j in range(self.size()):
            self._atoms[j].x += 1 if self._status == Status.DOWN_ALONG_SURFACE else -1
        self._define_rect_border()

    def merger(self, cluster) -> None:
        for atom in cluster._atoms:
            self.add_atom(atom)

        self._define_rect_border()

        if cluster.status == Status.LEAVING:
            self._status = Status.LEAVING

    def _symbol_at_relative(self, rel_i: int, rel_j: int) -> str:
        left = self.border_left()
        for k in range(self.size()):
            if self._atoms[k].x - left.x == rel_i and self._atoms[k].y - left.y == rel_j:
                return "*"
            if k == self.size() - 1:
                return " "
        return ""

    def image(self) -> str:
        image = ""
        for i in range(self.border_right().x - self.border_left().x + 1):
            for j in range(self.border_right().y - self.border_left().y + 1):
                image += self._symbol_at_relative(i, j)
            image += "\n"

        return image