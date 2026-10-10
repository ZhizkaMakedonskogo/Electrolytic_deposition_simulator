# TODO: is this class needed?

class Cell:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self._x = x  # i
        self._y = y  # j

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @x.setter
    def x(self, x: int):
        self._x = x

    @y.setter
    def y(self, y: int):
        self._y = y