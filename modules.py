#!/usr/bin/python
import numpy as np
import os
from geneticalgorithm2 import geneticalgorithm2 as ga
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity 
import shutil
import time
import concurrent.futures
from multiprocessing import Pool
import math
from smt.sampling_methods import LHS
from printing import *
from skopt import gp_minimize
from scipy.spatial import distance as sp




class Function():
    def __init__(self, train_data, indict):
        self.train_data = train_data
        self.indict = indict
        self.tree = BallTree(self.train_data, leaf_size=int(self.indict['leaf_size']), metric=self.indict['metric'])

    def _get_cos(self, idx, X):
        X = X.reshape(1, len(X))
        cos = np.zeros(len(idx[0]))
        for i, j in enumerate(idx[0]):
            cos[i] = cosine_similarity(self.train_data[j].reshape(1,-1),X.reshape(1,-1))
        cos_av = np.average(cos)
        return cos_av
    def _get_force(self, dist):
        '''
        COULOMB's LAW:
        F = k q1q2/r^2
        charge = q1q2 
        '''
        charge = float(self.indict['charge'])
        F = charge/(dist**2)
        F_av = np.average(F)
        return F_av

    def _get_radius(self, x):
        _xi = x*float(self.indict['xi'])
        x2 = _xi + x
        r2 = x2-x
        r = squrt(r2)
        return r
    def _run_external(self, X):
            try:
                from F import Fx
            except ImportError:
                raise ImportError('F.py was not found in /src directory')
            f = Fx(self.indict, self.train_data)
            return f.f_x(X)
        

    def f_x(self, X):

        X = np.array(X)

        dist, idx = self.tree.query(X, k=int(self.indict['k']))
        #First distace is 0 (to itself)
        dist = np.delete(dist, 0)
        idx = np.delete(idx, 0)
    
        if self.indict["f(x)"] == "BallTree_Force":
            return self._get_force(dist)

        elif self.indict["f(x)"] == "BallTree_COS":
            return self._get_cos(idx,X)

        elif self.indict["f(x)"] == "external":
            return self._run_external(X)
        else:
            print('WRONG f(x) keyword!')
            raise SystemExit



class CSPX_GA(Function):

    def __init__(self, indict, train_data):

        self.indict = indict
        self.train_data = train_data
        self.tree = BallTree(self.train_data, leaf_size=int(self.indict['leaf_size']), metric=self.indict['metric'])


    def run_GA(self, variables, fx=None):
         
        if fx == None:
            fx = self.f_x

        algorithm_param = {'max_num_iteration': float(self.indict['omptimisation_cycles']), 'population_size': float(self.indict['pop_size']),
                        'mutation_probability': float(self.indict['mut_prob']),
                           'elit_ratio': 0.01, 'crossover_probability': float(self.indict['cross_prob']),
                           'parents_portion': float(self.indict['parent_po']), 'crossover_type': 'uniform',
                           'max_iteration_without_improv': int(self.indict['max_iteration_without_improv'])}

        model = ga(function=fx, dimension=len(variables), variable_type='real',
                   variable_boundaries=variables, algorithm_parameters = algorithm_param)

        model.run(set_function= ga.set_function_multiprocess(fx, n_jobs=int(self.indict['n_processes'])), 
    no_plot=True,
    disable_progress_bar=True,
    disable_printing=True)

        return model.output_dict

class CSPX_GRID(Function):
    def __init__(self, indict, train_data):

        self.indict = indict
        self.train_data = train_data
        self.fx_values = np.zeros(int(self.indict["GRID_sample_number"]))
        self.tree = BallTree(self.train_data, leaf_size=int(self.indict['leaf_size']), metric=self.indict['metric'])

    def run_cspx_grid(self, variable_boundaries):

        if self.indict['random_seed'] != None:
                self.indict['random_seed'] = int((self.indict['random_seed']))

        sampling = LHS(xlimits=variable_boundaries, random_state=self.indict['random_seed'])
        points = sampling(int(self.indict["GRID_sample_number"]))
        if int(self.indict['n_processes']) > 1:

            pool = Pool(processes=int(self.indict['n_processes']))

            results = pool.map(self.f_x, points)
            pool.close()
            pool.join()

            for idx, x in enumerate(results):
                self.fx_values[idx] = x
                
        elif int(self.indict['n_processes']) == -1:
            pool = Pool(processes=os.cpu_count())
            results = pool.map(self.f_x, points)
            pool.close()
            pool.join()

            for idx, x in enumerate(results):
                self.fx_values[idx] = x


        elif int(self.indict['n_processes']) == 1:
    
            for idx, x in enumerate(points):
                self.fx_values[idx] = self.f_x(x)
        

        min_val_idx = np.argmin(self.fx_values, axis=0)
        

        return points[min_val_idx], self.fx_values[min_val_idx]


class CSPX_BO(Function):
    def __init__(self, indict, train_data):
        self.indict = indict
        self.train_data = train_data
        self.tree = BallTree(self.train_data, leaf_size=int(self.indict['leaf_size']), metric=self.indict['metric'])

    def run_bayassian(self, variable_boundaries):
        #Correction for boundary conditions as pg_minimize requires [(min, max),...]
        #hence for case min==max 1e-10 is added to max
        for i, j in enumerate(variable_boundaries):
            if variable_boundaries[i][1] == variable_boundaries[i][0]:
                variable_boundaries[i][1] = variable_boundaries[i][1] + 1e-10
        bounds = []
        for i, j in enumerate(variable_boundaries):
            bounds.append((variable_boundaries[i][0], variable_boundaries[i][1]))
        
        res = gp_minimize(self.f_x,                  # the function to minimize
                  bounds,      # the bounds on each dimension of x
                  acq_func="gp_hedge",      # the acquisition function
                  n_calls=int(self.indict['omptimisation_cycles']),         # the number of evaluations of f
                 n_random_starts=5, # the number of random initialization points
                         # the noise level (optional)
                  random_state=1234)   # the random seed
        '''
        print (res.fun, res.x)
        with open('reg_info.txt', 'w') as f:
            f.write(str(res))
            f.close()
        '''
        return res.x, res.fun

class Space():

    def __init__(self, indict):

        self.indict = indict
        with open(indict['in_file'], 'r', encoding='utf-8-sig') as f:
            self.data = np.genfromtxt(f, delimiter=',', dtype=float)
        self.max_bound = np.fromstring(self.indict["UBL"], sep=',')
        self.min_bound = np.fromstring(self.indict["LBL"], sep=',')

    def _boudary_conditions(self, X):

        """
        Correcting for boundary conditions
        !There might be problems with negative numbers!
        !Espetialy for GA algorithm as it expects:    !
        !lower_boundaries must be smaller than        !
        !upper_boundaries [lower,upper]               !
        """

        
        if self.indict["Apply_BD"] == "True":
            
            for i in range(len(self.max_bound)):
                #maximum boundaries
                if X[i][1] > self.max_bound[i]:
                    X[i][1] = self.max_bound[i]
                #Checks if max bound. doesn't have smaller number than in LBL and corrcets replacing with 
                #LBL value   
                if X[i][1] < self.min_bound[i]:
                    X[i][1] = self.min_bound[i]
                    
                #minimum boundaries
                if X[i][0] < self.min_bound[i]:
                    X[i][0] = self.min_bound[i]

                #Where UBL is equal to LBL
                if self.max_bound[i] == self.min_bound[i]:
                    X[i][0] = self.min_bound[i]
                    X[i][1] = self.max_bound[i]          

        elif self.indict["Apply_BD"] == "False":
            return X

        return X



    def sub_space(self):

       
        var_of_interest = self.data[self.data[:, -1] == 1] #Filters values of interest i.e. == 1 
        var_of_interest = var_of_interest[:, :-1]
    
        xi  = var_of_interest*float(self.indict['split_value'])

        plus_xi = []
        minus_xi = [] 
        
        for i in range(len(var_of_interest)):

            plus = var_of_interest[i] + xi[i]
            minus = var_of_interest[i] - xi[i]

            plus_xi.append(plus)
            minus_xi.append(minus)

        for i in range(len(minus_xi)):
             x = minus_xi[i]
             y = plus_xi[i]
             two_by_n = np.vstack((x, y)).T

        #Apply correction for boundary conditions

        two_by_n = self._boudary_conditions(two_by_n)
        

               
        return two_by_n
        
    def full_space(self):
        
        two_by_n = np.zeros((len(self.max_bound), 2))

        max_val = self.max_bound
        min_val = self.min_bound
        for i in range(len(max_val)):
            y = max_val[i]
            x = min_val[i]
            two_by_n[i] = np.vstack((x, y)).T 

        return two_by_n

    def sub_space_C(self):
        min_max = []

        u = self.indict["U"]
        l = self.indict["L"]

        max_val = np.fromstring(u, sep=',')
        min_val = np.fromstring(l, sep=',')

        two_by_n = np.vstack((min_val, max_val)).T
        
        min_max.append(two_by_n)
      
        return min_max

    def _sub_space_xi(self, data_point, xi):
        
        xi  = data_point*xi

        plus_xi = []
        minus_xi = [] 
        min_max = []
        two_by_n = np.zeros((len(data_point), 2))
    
        for i in range(len(data_point)):

            plus = data_point[i] + xi[i]
            minus = data_point[i] - xi[i]
            if minus > plus:
                plus, minus  = minus, plus

            plus_xi.append(plus)
            minus_xi.append(minus)


        for i in range(len(minus_xi)):
            x = minus_xi[i]
            y = plus_xi[i]
            two_by_n[i] = np.vstack((x, y)).T 

        #Apply correction for boundary conditions
        two_by_n = self._boudary_conditions(two_by_n)
         
        return two_by_n


class VOID(CSPX_GA, Space):
    '''
    Code explores voids in space based on number of neighbours around given data point (using radius).
    GA optimises until samallest number is reached. Calculated point is appended to training data set and 
    further point is calculated.
    '''
    def __init__(self, indict, train_data):
        self.indict = indict
        self.train_data = train_data
        with open(indict['in_file'], 'r', encoding='utf-8-sig') as f:
            self.data = np.genfromtxt(f, delimiter=',', dtype=float)
        self.max_bound = np.fromstring(self.indict["UBL"], sep=',')
        self.min_bound = np.fromstring(self.indict["LBL"], sep=',')
        self.points = self.full_space()
        self.f_x_radius_values = np.zeros(int(self.indict["sample_number"]))
        self.f_x_distances_values = np.zeros(int(self.indict["sample_number"]))

    def f_x_radius(self, X):
        X = X.reshape(1, len(X))
        tree = BallTree(self.train_data, leaf_size=int(self.indict['leaf_size']), metric=self.indict['metric'])
        n_neighbours = tree.query_radius(X, r=self.indict["r"], count_only=True) 
        return n_neighbours[0]

    def search(self):
        points = np.zeros((int(self.indict["sample_number"]), len(self.max_bound)))
        
        for i in range(int(self.indict["sample_number"])):
            start_time = time.time()

            optimised_point_dict = self.run_GA(self.points, self.f_x_radius)
            points[i] = optimised_point_dict['variable']
            
            self.train_data = np.vstack([self.train_data, optimised_point_dict['variable']])
            self.f_x_radius_values[i] = optimised_point_dict['function']

            end_time = time.time()
            void_loop_time = end_time - start_time


            print_void_info(i+1, int(self.f_x_radius_values[i]), void_loop_time)

        print(" Void search completed.\n")
        np.savetxt(f'{self.indict["out_dir"]}/void_search.csv', points, delimiter=",")

        return points







