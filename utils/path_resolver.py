from pathlib import Path

def construct_new_folder(path: Path):
    assert path.suffix == "", f"Path must be of a folder, not a filename {path}"
    new_path = path
    i = 0
    while new_path.exists():
        i += 1
        new_path = path.parent / (path.stem + str(i))

    return new_path
