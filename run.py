#!/usr/bin/python
import sys
from ChemSPX.main import Program


if __name__ == "__main__":
	try:
		inpt = sys.argv[1]
	except IndexError:
		print("Input file must be specified")
		raise SystemExit

	program = Program(inpt)
	program.run()