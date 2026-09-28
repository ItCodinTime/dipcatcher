"""Rewrite ``quality/broad_exceptions.txt`` from the current tree."""

from check_broad_exceptions import main

if __name__ == "__main__":
    raise SystemExit(main(["--write"]))
