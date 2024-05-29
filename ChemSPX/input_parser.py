import sys
import numpy as np


class InputParser:
    def __init__(self, inptfle):
        """
        Contains default values for input.
        """

        self.input = inptfle
        self.dict = {}
        self.default = {
            "print_parameters": "False",
            "init_data_sampling": "LHS",
            "out_dir": "OUT",
            "metric": "euclidean",
            "GRID_sample_number": 1000,
            "leaf_size": 20,
            "mut_prob": 0.1,
            "cross_prob": 0.5,
            "parent_po": 0.3,
            "elit_ratio": 0.01,
            "crossover_type": "uniform",
            "n_processes": 1,
            "write_f_every": 1,
            "split_value": 0.1,
            "max_iteration_without_improv": 50,
            "print_every": 1,
            "write_initial": "False",
            "map_function": "False",
            "random_seed": None,
            "check_conv_every": 10,
            "power": 1,
            "k": "all",
            "h": 0.1,
            "verbose": 1,
            "ploop": "False",
            "PCA": "False",
            "pca_n_components": 2,
        }

    def _check_indict(self):
        """
        Check if neccessary parameters are defined in input file
        and if not default values are inserted.
        """

        important = [
            "OPT_method",
            "init_data_sampling",
            "Apply_BD",
            "UBL",
            "LBL",
            "sample_number",
            "iteration_num",
            "method",
            "xi",
            "f(x)",
        ]

        try:
            for key in important:
                if key not in self.dict:
                    print(f"ERROR: {key} mus be defined in program input file!")
                    raise SystemExit
        except:
            raise SystemExit

        for key, value in self.default.items():
            if key not in self.dict:
                self.dict[key] = value

        # Corrects true/false infut
        for key, value in self.dict.items():
            if value == "true" or value == "t" or value == "T":
                self.dict[key] = "True"
            elif value == "false" or value == "f" or value == "F":
                self.dict[key] = "False"

        self._check_input_ref_fle()

    def get(self) -> dict:

        if type(self.input) == str:

            f = open(self.input, "r")

            for line in f:
                if not line.startswith("#"):
                    if not line == "\n":
                        key = line.split(" ", 1)[0].strip()
                        val = line.split(" ", 1)[1].strip()
                        if "#" in val:
                            val = val.split("#", 1)[0].strip()
                        self.dict[key] = val

        elif type(self.input) == dict:
            for key, element in self.input.items():
                self.dict[key] = element

        else:
            print("ERROR: Could not parse input file. Use .csv file.")
            raise SystemExit
        self._check_indict()

        return self.dict

    def _check_input_ref_fle(self):
        with open(self.dict["in_file"], "r", encoding="utf-8-sig") as f:
            train_data = np.genfromtxt(f, delimiter=",", dtype=float)

        if np.isnan(np.sum(train_data)):
            print("ERROR: Referece data input contains NaN values!")

        if sum(train_data[:, -1]) != 0:
            print(
                "ERROR: Last column in the reference data file must conatin 0 or 1 values!"
            )
            raise SystemExit


if __name__ == "__main__":
    inpt = InputParser(sys.argv[1])
    inpt.get()
    inpt._check_indict()
