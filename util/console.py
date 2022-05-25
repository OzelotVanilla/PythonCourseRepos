import operator
import os
import time
from functools import reduce


class console:
    # Helper class

    def info(*args, color_rgb="0094c8", sep="") -> None:
        print(console.__getColorANSICodeFromHexRGB(color_rgb) + "[INFO] ", end="")
        print(*args, "\033[39m", sep=sep)

    def warn(*args, color_rgb="fcc800", sep="") -> None:
        print(console.__getColorANSICodeFromHexRGB(color_rgb) + "[WARN] ", end="")
        print(*args, "\033[39m", sep=sep)

    def err(*args, color_rgb="ba2636", sep="") -> None:
        print(console.__getColorANSICodeFromHexRGB(color_rgb) + "[ERR!] ", end="")
        print(*args, "\033[39m", sep=sep)

    def clear() -> None:
        print("\033c")

    def wait(seconds: float) -> None:
        time.sleep(seconds)

    def getWidth() -> int:
        return os.get_terminal_size().columns

    def getHeight() -> int:
        return os.get_terminal_size().lines

    def __getColorANSICodeFromHexRGB(color_rgb: str) -> str:
        if color_rgb[0] == "#":
            color_rgb = color_rgb[1:]

        return "\033[38;2" + reduce(
            operator.add, [";" + str(int(color_rgb[i:i + 2], base=16)) for i in [0, 2, 4]]
        ) + "m"
