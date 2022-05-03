from util.console import console
from util.helper import executePrelude, executeFinale

# Use VSCode to open the entire folder, then run this script
# Otherwise, the import may not be solved


def main():
    # Do preparation jobs, like install tools, download datasets to specified path
    # Do not worry, almost all config would be clean-up if you want
    executePrelude()

    # Finished
    executeFinale()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stop the program manually.")
        executeFinale()
