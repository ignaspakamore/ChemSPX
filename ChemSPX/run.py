#!/usr/bin/env python3
import sys
import argparse
from ChemSPX import ChemSPX


def program():
    try:
        input = sys.argv[1]
    except IndexError:
        print("ERROR: Input file must be specified.")
        raise SystemExit

    ChemSPX(input).run()


if __name__ == "__main__":
    program()
