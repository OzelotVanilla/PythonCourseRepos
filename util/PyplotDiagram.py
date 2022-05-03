from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.image as imgplt

# Use pyplot.show to show all figures


class PyplotDiagram:
    count_num = 0

    class PlotType(Enum):
        pending = 0
        function = 1
        image = 2
        xy_data = 3
    # self.plot_type = enum { image, function, xy_data, pending }

    def __init__(self):
        plt.figure(PyplotDiagram.count_num)
        self.id = PyplotDiagram.count_num
        PyplotDiagram.count_num += 1
        self.plot_type = PyplotDiagram.PlotType.pending

    def loadImage(self, path: str):
        if self._checkIfAbleToAdd():
            plt.figure(self.id)
            plt.imshow(imgplt.imread(path))
            return self
        else:
            self._failToAdd()

    def loadFunction(self, func):
        plt.figure(self.id)

    def addTitle(self, title: str):
        plt.figure(self.id)
        plt.title(title)

    def _checkIfAbleToAdd(self): return self.plot_type == self.PlotType.pending

    def _showFailToAddMessage(): pass
