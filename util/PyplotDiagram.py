from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.image as imgplt

from util.console import console

# Use pyplot.show to show all figures


class PyplotDiagram:
    count_num = 1

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

    def addAsSeries(self, data: dict[str, dict[str, float]], width: float = 0.2, show_legend: bool = True) -> None:
        # data should be like: {"2015": {"a": 1, "b": 2}, "2020": {"a": 4, "b": 5}}
        if not self.__checkIfAbleToAdd():
            console.warn("Failed to do addAsSeries because there was already data")
        plt.figure(self.id)

        # Add data to the set
        data_names, data_labels = set(), set()
        for data_name in data.keys():
            data_names.add(data_name)
            [data_labels.add(label_name) for label_name in data[data_name].keys()]

        # Calculating the width, each bunch of series has
        bar_position = [i * width for i in len(data_labels)]

        pass

    def addTitle(self, title: str, /, ):
        plt.figure(self.id)
        plt.title(title)

    def showAllPlot():
        plt.show()

    def __checkIfAbleToAdd(self): return self.plot_type == self.PlotType.pending

    def __showFailToAddMessage(): pass

    def __failToAdd(): pass
