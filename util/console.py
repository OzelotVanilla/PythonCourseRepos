import os
import time


class console:
    # Helper class

    def info(*args, color_rgb="0094c8", sep="") -> None:
        print(console._getColorANSICodeFromHexRGB(color_rgb), "[INFO] ", *args, "\033[39m", sep=sep)

    def warn(*args, color_rgb="fcc800", sep="") -> None:
        print(console._getColorANSICodeFromHexRGB(color_rgb), "[WARN] ", *args, "\033[39m", sep=sep)

    def err(*args, color_rgb="ba2636", sep="") -> None:
        print(console._getColorANSICodeFromHexRGB(color_rgb), "[ERR!] ", *args, "\033[39m", sep=sep)

    def clear() -> None:
        print("\033c")

    def wait(seconds: float) -> None:
        time.sleep(seconds)

    def getWidth() -> int:
        return os.get_terminal_size().columns

    def getHeight() -> int:
        return os.get_terminal_size().lines

    def _getColorANSICodeFromHexRGB(color_rgb: str) -> str:
        if color_rgb[0] == "#":
            color_rgb = color_rgb[1:]
        color_decimal = [";" + str(int(color_rgb[i:i + 2], base=16)) for i in [0, 2, 4]]
        color_code = "\033[38;2"
        for c in color_decimal:
            color_code = color_code + c
        return color_code + "m"
