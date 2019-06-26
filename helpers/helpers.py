import os


def all_paths_in_dir(path, file_type=".txt"):
    files = []
    for r, d, f in os.walk(path):
        for file_path in f:
            if file_type in file_path:
                files.append(os.path.join(r, file_path))
    if not files:
        print("No {} files in {}, is it correct path?".format(file_type, path))
    return files
