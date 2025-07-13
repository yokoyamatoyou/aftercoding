import pathlib
import py_compile
import sys

for path in pathlib.Path('.').rglob('*.py'):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as e:
        print(f'Failed to compile {path}: {e}', file=sys.stderr)
        sys.exit(1)
