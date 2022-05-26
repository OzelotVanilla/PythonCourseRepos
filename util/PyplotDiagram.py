import math
from util.console import console

from enum import Enum
import matplotlib.pyplot as plt


class PyplotDiagram:
    ...
# Use pyplot.show to show all figures


class PyplotDiagram:
    count_num = 1

    class PlotType(Enum):
        pending = 0
        series = 1
        drawn = -1
    # self.plot_type = enum { image, function, xy_data, pending }

    def __init__(self):
        plt.figure(PyplotDiagram.count_num)
        self.id = PyplotDiagram.count_num
        PyplotDiagram.count_num += 1
        self.plot_type = PyplotDiagram.PlotType.pending

    def addAsSeries(self, data: dict[str, dict[str, float]],
                    /, draw_now: bool = False, allow_overwrite: bool = True,
                    width: float = 0.2) -> PyplotDiagram:
        # data should be like: {"2015": {"a": 1, "b": 2}, "2020": {"a": 4, "b": 5}}
        if (not self.__checkIfAbleToAdd()) and self.plot_type != PyplotDiagram.PlotType.series:
            self.__showFailToAddMessage()
            return
        else:
            self.plot_type = PyplotDiagram.PlotType.series

        # Create dict if not exist
        if not hasattr(self, "data"):
            self.data = dict()

        # Add new data to existing data
        for key in data.keys():
            if (not allow_overwrite) and (key in self.data):
                console.warn("Key \"" + str(key) + "\" already exists, ignored.")
                continue
            else:
                for subkey in data[key].keys():
                    if (not allow_overwrite) and (subkey in self.data.get(key, dict())):
                        console.warn("Sub-Key \"" + str(key) + "\" for key \"" + key + "\"already exists, ignored.")
                        continue
                    else:
                        self.data[key] = {**self.data.get(key, dict()), **data[key]}

        if draw_now:
            return self.drawSeries(width)
        else:
            return self

    def drawSeries(self, /,
                   width: float = 0.2, interval: float = 0.2,
                   show_value: bool = True, show_legend: bool = True, show_double_axis: bool = False,
                   y_left_for: str = None, y_right_for: str = None,
                   scale_y_left: float = None, scale_y_right: float = None) -> PyplotDiagram:

        plt.figure(self.id)

        # Get two axis, because code after always use that
        y_left = plt.gca()
        y_right = y_left.twinx()
        plt.sca(y_left)

        # data should be like: {"2015": {"a": 1, "b": 2}, "2020": {"a": 4, "b": 5}}
        # Add data to the set
        data = self.data
        data_names, data_labels = [], []
        for data_name in data.keys():
            data_names.append(data_name) if data_name not in data_names else None
            for label_name in data[data_name].keys():
                data_labels.append(label_name) if label_name not in data_labels else None

        # Each time, draw in data like "2015": {"a": 1, "b": 2}
        nth_data = 0
        for data_name in data_names:
            # For each data, to keep in same legend, draw them at once
            bar_position = [
                nth_data * width + (len(data_names) * width + interval) * i for i in range(len(data_labels))
            ]
            # If there is no data can be queried, use 0 as default
            values = [data[data_name].get(label, 0) for label in data_labels]
            plt.bar(
                bar_position, values,
                width=width, align="edge", label=data_name
            )
            nth_data += 1

            # Show the value of bar
            if show_value:
                for x, y in zip(bar_position, values):
                    plt.text(x + width / 2, y, f"{y:.4}" if type(y) == float else y, ha="center", va="bottom")

        del nth_data

        # Draw series labels
        plt.xticks(
            [len(data_names) * width / 2 + (len(data_names) * width + interval) * i for i in range(len(data_labels))],
            data_labels
        )

        # Show legend if required
        if show_legend:
            y_left.legend(loc="upper left")

        # Show double axis
        if show_double_axis:
            y_right.set_yticks(y_left.get_yticks())

        return self

    def setTitle(self, title: str, /, ) -> PyplotDiagram:
        plt.figure(self.id)
        plt.title(title)
        return self

    def clear(self) -> PyplotDiagram:
        plt.figure(self.id)
        plt.clf()
        self.plot_type = PyplotDiagram.PlotType.pending
        return self

    def showAllPlot():
        console.info("Showing figures, close figures window to continue on program.")
        plt.show()

    def __checkIfAbleToAdd(self): return self.plot_type == PyplotDiagram.PlotType.pending

    def __showFailToAddMessage(self):
        console.warn(
            "The plot No.", self.count_num, "failed to add diagram because it already has.",
            "Use \"clear\" method to clear the figure first, then add new diagram.", sep=" "
        )
