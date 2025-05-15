import sys
import numpy as np


class InputParser:
    def __init__(self, inptfle):
        """
        Contains default values for input.
        """

        self.input = inptfle
        self.indict = {}
        self.default = {
            "print_parameters": "False",
            "init_data_sampling": "LHS",
            "out_dir": "OUT",
            
            "metric": "euclidean",
            "GRID_sample_number": 1000,
            "leaf_size": 20,

            # GA optimiser parameters
            "OPT_method": "GA",
            "mut_prob": 0.1,
            "cross_prob": 0.5,
            "parent_po": 0.3,
            "elit_ratio": 0.01,
            "split_value": 0.1,
            "crossover_type": "uniform",
            "max_iteration_without_improv": 50,

            "n_processes": 1,
            "write_f_every": 1,


            "print_every": 1,
            "verbose": 1,
            "random_seed": 45, 

            "write_initial": "False",
            "map_function": "False",
            "check_conv_every": 10,
            "power": 1,
            "k": "all",
            "h": 0.1,
            "PCA": "False",
            "pca_n_components": 2,
            "f(x)":"Force",

        }

    def _check_indict(self):
        """
        Check if neccessary parameters are defined in input file
        and if not default values are inserted.
        """

        important = [
            "init_data_sampling",
            "Apply_BD",
            "UBL",
            "LBL",
            "sample_number",
            "iteration_num",
            "method",
            "xi",
        ]

        try:
            for key in important:
                if key not in self.indict:
                    print(f"ERROR: {key} must be defined in program input fil!")
                    raise SystemExit
        except KeyError:
            raise SystemExit

        for key, value in self.default.items():
            if key not in self.indict:
                self.indict[key] = value

        # Corrects true/false input
        for key, value in self.indict.items():
            if value == "true" or value == "t" or value == "T" or (type(value) == bool and value == True):
                self.indict[key] = "True"
            elif value == "false" or value == "f" or value == "F" or (type(value) == bool and value == False):
                self.indict[key] = "False"

        self._check_input_ref_fle()

    def get(self) -> dict:

        if isinstance(self.input, str):

            f = open(self.input, "r")

            for line in f:
                if not line.startswith("#"):
                    if not line == "\n":
                        key = line.split(" ", 1)[0].strip()
                        val = line.split(" ", 1)[1].strip()
                        if "#" in val:
                            val = val.split("#", 1)[0].strip()
                        self.indict[key] = val

        elif type(self.input) == dict:
            for key, element in self.input.items():
                self.indict[key] = element

        else:
            print("ERROR: Could not parse input file. Use .csv file.")
            raise SystemExit
        self._check_indict()

        return self.indict

    def read_input_file(self):
        if self.indict["init_data_sampling"] != "LHSEQ":
            try:
                with open(self.indict["in_file"], "r", encoding="utf-8-sig") as f:
                    self.data_points = np.genfromtxt(f, delimiter=",")
                    # Remove headers

                    if np.all(np.isnan(self.data_points[0])):
                        self.data_points = self.data_points[1:, :]

                    self.data_points = self.data_points.astype("float64")
            except FileNotFoundError:
                print(f"ERROR: {self.indict["in_file"]} file was not found.")
                raise SystemExit
                
        elif self.indict["init_data_sampling"] == "LHSEQ":
            self.data_points = None

    def _check_input_ref_fle(self):

        self.read_input_file()

        if np.isnan(np.sum(self.data_points)):
            print("ERROR: Referece data input contains NaN values!")

        if (
            np.all((self.data_points[:, -1] == 0) | (self.data_points[:, -1] == 1))
            is False
        ):
            print(
                "ERROR: Last column in the reference data file must contain 0 or 1 values!"
            )

            raise SystemExit


if __name__ == "__main__":
    inpt = InputParser(sys.argv[1])
    inpt.get()
    inpt._check_indict()
