#!/usr/bin/python
from ChemSPX.functions import (
    CSPX_GA,
    CSPX_GRID,
    CSPX_BO,
    VOID,
    Space,
    Function,
    RandomSample,
)
from ChemSPX.pca import PCA
from ChemSPX.printing import (
    _print_logo,
    _print_pars,
    _print_finished,
    _print_loop_info,
    _print_init_info,
    _print_loop_conv,
)
from ChemSPX.input_parser import InputParser
from smt.sampling_methods import LHS
from sklearn.neighbors import BallTree
from multiprocessing import Pool
import pandas as pd
from scipy import stats
import numpy as np
import itertools
import shutil
import time
import sys
import os


class ChemSPX:
    def __init__(self, input: str | dict) -> None:
        """
        The 'ChemSPX' class is the core of the ChemSPX Python package,
        cntaining key calculation functions and the primary 'run' operation.
        """

        self.indict = InputParser(input).get()

        if self.indict["init_data_sampling"] != "LHSEQ":
            with open(self.indict["in_file"], "r", encoding="utf-8-sig") as f:
                self.train_data = np.genfromtxt(f, delimiter=",", dtype=float)
                self.train_data = self.train_data[:, :-1]
                self.train_size = len(self.train_data)
        else:
            self.train_data = None

        self.fx1 = np.zeros(int(self.indict["sample_number"]))
        self.fx2 = np.zeros(int(self.indict["sample_number"]))
        self.av_del_fx = 0
        self.std_fx = 0
        self.del_fx = 0
        self.der_fx1 = 0
        self.der_fx2 = 0
        self.av_fx = 0
        self.del_vector = 0
        self.xi = float(self.indict["xi"])
        self.xi_init = float(self.indict["xi"])
        self.step = int(self.indict["sample_number"])
        self.vect_change = np.zeros(int(self.indict["sample_number"]))
        self.max_bound = [float(x) for x in np.fromstring(self.indict["UBL"], sep=",")]
        self.min_bound = [float(x) for x in np.fromstring(self.indict["LBL"], sep=",")]

    def _get_space_var(self) -> list:
        """
        Returns:
            Space variable bounderies (min, max values) for each data vector.
        """

        if self.indict["method"] == "full_space":
            space_variables = Space(self.indict).full_space()
        elif self.indict["method"] == "sub_space_C":
            space_variables = Space(self.indict).sub_space_C()
        elif self.indict["method"] == "sub_space":
            space_variables = Space(self.indict).sub_space()

        return space_variables

    def _get_initial_fx(self, points: list) -> None:
        """
        Args:
            ...
        Returns:
            ...
        """

        if int(self.indict["n_processes"]) == 1:
            for i, point in enumerate(points):
                fx = Function(self.train_data, self.indict).f_x(point)
                self.fx1[i] = fx
        elif int(self.indict["n_processes"]) > 1:

            pool = Pool(processes=int(self.indict["n_processes"]))
            results = pool.map(Function(self.train_data, self.indict).f_x, points)
            for i in range(len(results)):
                self.fx1[i] = results[i]
            pool.close()
            pool.join()
        elif int(self.indict["n_processes"]) == -1:
            pool = Pool(processes=os.cpu_count())
            results = pool.map(Function(self.train_data, self.indict).f_x, points)
            for i in range(len(results)):
                self.fx1[i] = results[i]
            pool.close()
            pool.join()

    def _generate_grid_coordinates(self, step) -> list:
        """
        Generate vertex coordinates of an n-dimensional grid with custom minimum and maximum values.

        Parameters:
            min_values (list): List of minimum values for each dimension.
            max_values (list): List of maximum values for each dimension.
            step (float): Step size for each dimension.

        Returns:
            list: List of vertex coordinates.
        """

        # Generate ranges for each dimension
        ranges = [
            np.arange(min_val, max_val + step, step)
            for min_val, max_val in zip(self.min_bound, self.max_bound)
        ]

        # Generate all possible combinations of indices for each dimension
        indices = list(itertools.product(*ranges))
        indices = [list(i) for i in indices]

        print(f"NOTE: Function distribution grid consists of {len(indices)} points.")

        return indices

    def _eval_fx_distribution(self) -> None:
        """
        Args:
            ...
        """

        points = self._generate_grid_coordinates(float(self.indict["map_grid_size"]))

        function = Function(self.train_data, self.indict)

        # Compute fx values of grid points in parallel
        pool = Pool(int(self.indict["n_processes"]))
        fx = pool.map(function.f_x, points)

        kde = stats.gaussian_kde(fx)
        x_vals = np.linspace(min(fx), max(fx), 100)
        result = kde.evaluate(x_vals)
        result = np.c_[x_vals, result]

        f = open(f'{self.indict["out_dir"]}/fx_map.csv', "a")
        np.savetxt(f, result, delimiter=",", fmt="%s")
        f.close()

        print("DONE: Function distribution calculated.")

    def _get_initial_stats(self) -> None:
        self.av_del_fx = np.average(self.fx1)
        self.std_fx = np.std(self.fx1)
        if self.indict["map_function"] == "False":
            _print_init_info(self.av_del_fx, self.std_fx, len(self.fx1))

    def _get_stats(self) -> None:
        self.del_fx = (self.fx2 - self.fx1) ** 2
        self.av_del_fx = np.average(self.del_fx)
        self.std_fx = np.std(self.del_fx)

    def _get_vect_change(self, x1: list, x2: list) -> list:
        # x1 = x1/np.linalg.norm(x1)
        # x2 = x2/np.linalg.norm(x2)

        delX = np.subtract(x1, x2)
        vect_mag = np.linalg.norm(delX)

        return vect_mag

    def _check_convergence(self) -> int:

        cond1 = "NO"
        cond2 = "NO"
        cond3 = "NO"

        if self.av_fx <= float(self.indict["conv_fx"]):
            cond1 = "YES"
        if self.av_del_fx <= float(self.indict["conv_del_fx"]):
            cond2 = "YES"
        if self.del_vector <= float(self.indict["conv_vec"]):
            cond3 = "YES"
        _print_loop_conv(cond1, cond2, cond3)

        if cond1 == cond2 == cond3 == "YES":
            print("DONE: Converged on all three criteria.")
            return 0
        else:
            return 1

    def _print_data_table(self) -> None:

        print("\n")
        for i in range(len(self.train_data)):
            print(f"  {i} {self.train_data[i][:]}")
        print("\n")

    def _initial_sampling(self) -> None:

        if self.indict["init_data_sampling"] == "LHS":
            variable_bounderies = self._get_space_var()
            # Latin hypercube sampling
            if self.indict["random_seed"] is not None:
                self.indict["random_seed"] = int((self.indict["random_seed"]))
            sampling = LHS(
                xlimits=variable_bounderies, random_state=self.indict["random_seed"]
            )
            points = sampling(int(self.indict["sample_number"]))

        elif self.indict["init_data_sampling"] == "LHSEQ":
            variable_bounderies = self._get_space_var()
            # Latin hypercube sampling
            if self.indict["random_seed"] is not None:
                self.indict["random_seed"] = int((self.indict["random_seed"]))
            sampling = LHS(
                xlimits=variable_bounderies, random_state=self.indict["random_seed"]
            )
            points = sampling(int(self.indict["sample_number"]))
            self.train_data = points

        elif self.indict["init_data_sampling"] == "void":
            # VOID exploration algorithm
            sampling = VOID(self.indict, self.train_data)
            points = sampling.search()

        elif self.indict["init_data_sampling"] == "restart":
            # Read in data points for restart of calculation:
            f = self.indict["restart_file_name"]
            points = np.genfromtxt(f, delimiter=",", dtype=float)
            self.indict["sample_number"] = len(points)
            print(f"NOTE: Restart data taken from {f} file.")

        elif self.indict["init_data_sampling"] == "random":
            variable_bounderies = self._get_space_var()

            random = RandomSample(
                n_samples=int(self.indict["sample_number"]),
                boundaries=variable_bounderies,
                random_seed=int(self.indict["random_seed"]),
            )

            points = random.sample_n_dimensional_space()

        else:
            print(self.indict["init_data_sampling"])
            print("ERROR: Wrong initial sampling method specified.")
            raise SystemExit

        if self.indict["map_function"] == "True":
            self._eval_fx_distribution()

        # If True: does not combine points with reference data, hence f(x) is calculated only to the respect to ref data
        # and not generated sample points allowing to map function.
        if (
            self.indict["map_function"] == "False"
            and self.indict["init_data_sampling"] != "LHSEQ"
        ):
            self.train_data = np.vstack((self.train_data, points))

        self._get_initial_fx(points)

        # Write initial data out:
        if self.indict["write_initial"] == "True":
            np.savetxt(
                f"{self.indict['out_dir']}/initial_points.csv", points, delimiter=","
            )
            np.savetxt(
                f"{self.indict['out_dir']}/initial_fx.csv", self.fx1, delimiter=","
            )

    def _optimisation_loop(self) -> None:
        """
        Comment
        """
        if self.indict["init_data_sampling"] == "LHSEQ":
            self.train_size = 0

        for itt in range(int(self.indict["iteration_num"])):
            start_time_loop = time.time()

            for ix in range(int(self.indict["sample_number"])):
                point_idx = ix + self.train_size
                point = self.train_data[point_idx]

                # print(np.where(self.train_data == point))

                # generate point boundaries
                point_bounderies = Space(self.indict)._sub_space_xi(point, self.xi)

                if self.indict["OPT_method"] == "GA":
                    optimised_point_dict = CSPX_GA(self.indict, self.train_data).run_GA(
                        point_bounderies
                    )
                    optimised_point = optimised_point_dict["variable"]
                    f_x = optimised_point_dict["score"]

                elif self.indict["OPT_method"] == "GRID":
                    optimised = CSPX_GRID(self.indict, self.train_data).run_cspx_grid(
                        point_bounderies
                    )
                    optimised_point = optimised[0]
                    f_x = optimised[1]

                elif self.indict["OPT_method"] == "BO":
                    optimised = CSPX_BO(self.indict, self.train_data).run_bayassian(
                        point_bounderies
                    )
                    optimised_point = optimised[0]
                    f_x = optimised[1]

                else:
                    print("ERROR: Wrong optimisation method specified!")
                    raise SystemExit

                x1 = point  # x1 and x2 for vector diff.
                x2 = optimised_point
                self.vect_change[ix] = self._get_vect_change(x1, x2)

                self.fx2[ix] = f_x
                self.train_data[point_idx] = optimised_point

            # self._print_data_table()

            self._get_stats()
            # vector change
            self.del_vector = np.average(self.vect_change)
            self.av_fx = np.average(self.fx1)

            end_time_loop = time.time()
            loop_time = end_time_loop - start_time_loop

            # Calculation of the derrivative of f(x)
            self.der_fx2 = self.av_del_fx
            der_fx = self.der_fx1 - self.der_fx2
            self.der_fx1 = self.der_fx2

            if itt == 0:
                der_fx = 0

            _print_loop_info(
                itt + 1,
                self.av_fx,
                self.av_del_fx,
                self.del_vector,
                loop_time,
                self.indict["OPT_method"],
                self.indict["print_every"],
            )
            self.fx1 = self.fx2
            self.fx2 = np.zeros(int(self.indict["sample_number"]))

            if (itt + 1) % int(self.indict["write_f_every"]) == 0:

                np.savetxt(
                    f'{self.indict["out_dir"]}/iteration_{itt+1}.csv',
                    self.train_data[self.train_size : len(self.train_data)],
                    delimiter=",",
                )

                # PCA reduction
                if self.indict["PCA"] == "True":
                    PCA(f'{self.indict["out_dir"]}/iteration_{itt+1}.csv').reduce()

            # Writes out stats data:
            # Average derrivative of f(x), average of f(x), 2nd derrivative of average of f(x), std of average f(x), and loop time
            # ---------------------------  ---------------  ---------------   ----------------  -------------------      ---------

            if itt == 0:
                fx_header = np.array(
                    [
                        [
                            "iteration",
                            "average of f(x)",
                            "Average derrivative of f(x)",
                            "2nd derrivative of average of f(x)",
                            "std of average f(x)",
                            "average vec. change",
                            "loop time",
                        ]
                    ]
                )
                fx_data = np.array(
                    [
                        [
                            itt + 1,
                            self.av_fx,
                            self.av_del_fx,
                            der_fx,
                            self.std_fx,
                            self.del_vector,
                            loop_time,
                        ]
                    ]
                )

                f = open(f'{self.indict["out_dir"]}/fx_data.csv', "a")
                np.savetxt(f, fx_header, delimiter=",", fmt="%s")
                np.savetxt(f, fx_data, delimiter=",")
                f.close()

            else:
                fx_data = np.array(
                    [
                        [
                            itt + 1,
                            self.av_fx,
                            self.av_del_fx,
                            der_fx,
                            self.std_fx,
                            self.del_vector,
                            loop_time,
                        ]
                    ]
                )
                f = open(f'{self.indict["out_dir"]}/fx_data.csv', "a")
                np.savetxt(f, fx_data, delimiter=",")
                f.close()
            if (itt + 1) % int(self.indict["check_conv_every"]) == 0:
                convergence = self._check_convergence()
                if convergence != 0:
                    pass
                else:
                    break

    def run(self) -> None:
        """
        Funtion description
        """

        start_time = time.time()

        if int(self.indict["verbose"]) != 0:
            _print_logo()
        if (
            int(self.indict["n_processes"]) != -1
            and int(self.indict["n_processes"]) != 1
        ):
            print(f'NOTE: Number of processes set to {self.indict["n_processes"]}.')
        elif int(self.indict["n_processes"]) == -1:
            print(f"NOTE: Number of processes set to (-1) {os.cpu_count()}.")

        if self.indict["print_parameters"] == "True":
            _print_pars(self.indict)

        if os.path.exists(self.indict["out_dir"]):
            shutil.rmtree(self.indict["out_dir"])

        os.makedirs(self.indict["out_dir"])

        self._initial_sampling()
        self._get_initial_stats()
        self._optimisation_loop()

        if int(self.indict["verbose"]) != 0:
            _print_finished()

        end_time = time.time()
        run_time = end_time - start_time

        print(f"FINISHED: Time elapsed {run_time:.5} s")
