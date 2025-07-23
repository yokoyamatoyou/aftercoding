import pathlib
import py_compile
import sys

# Always search from this repository's root directory so running the script from
# another working directory won't accidentally traverse the Python installation
# or unrelated folders.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

for path in ROOT_DIR.rglob('*.py'):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as e:
        print(f'Failed to compile {path}: {e}', file=sys.stderr)
        sys.exit(1)
