"""Ensure directories exist before running the project."""

from pathlib import Path

if __name__ == "__main__":
    # Ensure the directories exist
    if not Path("src/classifier/out").exists():
        msg = ("The directory src/classifier/out does not exist. "
            + "Please run `mkdir src/classifier/out` to make the directory, "
            + "or run `git clone --recurse-submodules` if you "
            + "want to load the results already calculated."
            )
        raise FileNotFoundError(msg)
    Path("./graphics/out").mkdir(parents=True, exist_ok=True)
    Path("./out/cross_validation").mkdir(parents=True, exist_ok=True)
