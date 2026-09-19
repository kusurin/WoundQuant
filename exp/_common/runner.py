"""Invoke experiment command-line functions using explicit arguments."""
import sys

def call(main, arguments):
    old = sys.argv
    sys.argv = [old[0], *map(str, arguments)]
    try:
        return main()
    finally:
        sys.argv = old

