from util.console import console
from util.helper import executePrelude


def main():
    executePrelude()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stop the program manually.")
