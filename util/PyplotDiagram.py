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

    def addAsSeries(self, data: dict[str, dict[str, float]], /,
                    width: float = 0.2, interval: float = 0.2, show_legend: bool = True) -> None:
        # data should be like: {"2015": {"a": 1, "b": 2}, "2020": {"a": 4, "b": 5}}
        if not self.__checkIfAbleToAdd():
            console.warn("Failed to do addAsSeries because there was already data")
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
            bar_position = [
                nth_data * width + (len(data_names) * width + interval) * i for i in range(len(data_labels))
            ]
            plt.bar(
                bar_position,
                [data[data_name].get(label, 0) for label in data_labels],
                width=width,
                align="edge",
                label=data_name
            )
            nth_data += 1
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

    def showAllPlot():
        plt.show()

    def __checkIfAbleToAdd(self): return self.plot_type == self.PlotType.pending

    def __showFailToAddMessage(): pass

    def __failToAdd(): pass
