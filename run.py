#!/usr/bin/python
import sys
from ChemSPX import ChemSPX


if __name__ == "__main__":
    try:
        inpt = sys.argv[1]
    except IndexError:
        print("ERROR: Input file must be specified.")
        raise SystemExit

    program = ChemSPX(inpt).run()
