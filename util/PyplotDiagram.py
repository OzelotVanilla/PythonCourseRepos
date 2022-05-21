from util.console import console

from enum import Enum
import matplotlib.pyplot as plt


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

    def addAsSeries(self, data: dict[str, dict[str, float]], /,
                    width: float = 0.2, interval: float = 0.2,
                    show_value: bool = True, show_legend: bool = True) -> None:
        # data should be like: {"2015": {"a": 1, "b": 2}, "2020": {"a": 4, "b": 5}}
        if not self.__checkIfAbleToAdd():
            self.__showFailToAddMessage()
            return

        plt.figure(self.id)

        # Add data to the set
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
                bar_position,
                values,
                width=width,
                align="edge",
                label=data_name
            )
            nth_data += 1

            # Show the value of bar
            if show_value:
                for x, y in zip(bar_position, values):
                    plt.text(x + width / 2, y, f"{y:.4}", ha="center", va="bottom")

        del nth_data

        # Draw series labels
        plt.xticks(
            [len(data_names) * width / 2 + (len(data_names) * width + interval) * i for i in range(len(data_labels))],
            data_labels
        )

        # Show legend if required
        plt.legend(loc="upper left") if show_legend else None

    def addTitle(self, title: str, /, ):
        plt.figure(self.id)
        plt.title(title)

    def clear(self):
        plt.figure(self.id)
        plt.clf()
        self.plot_type = PyplotDiagram.PlotType.pending

    def showAllPlot():
        plt.show()

    def __checkIfAbleToAdd(self): return self.plot_type == PyplotDiagram.PlotType.pending

    def __showFailToAddMessage(self):
        console.warn(
            "The plot No.", self.count_num, " failed to add diagram because it already has.",
            "Use \"clear\" method to clear the figure first, then add new diagram."
        )
