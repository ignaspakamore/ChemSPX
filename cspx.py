from modules import CSPX_GA, CSPX_GRID, CSPX_BO, VOID, Space, Function
from pca import PCA
from printing import *
from datetime import datetime
import time
import numpy as np
import csv
import os
import sys
import shutil
from input_parser import InputParser
import math
from smt.sampling_methods import LHS
from multiprocessing import Pool


    

class CSPX:
    def __init__(self, input):

        '''
        Training data np.array() is kept in list[]. Array of index 0 is used, then new computed array is appended as idx 1.
        The old array is then deleted, 1 becomes 0 and the loop continues...
        '''
        self.inp = InputParser(input)
        self.indict = self.inp.get()
        with open(self.indict['in_file'], 'r', encoding='utf-8-sig') as f:
            self.train_data = np.genfromtxt(f, delimiter=',', dtype=float)
            self.train_data = self.train_data[:, :-1]
        self.fx1 = np.zeros(int(self.indict["sample_number"]))
        self.fx2 = np.zeros(int(self.indict["sample_number"]))
        self.av_del_fx = 0
        self.std_fx = 0
        self.del_fx = 0
        self.der_fx1 = 0
        self.der_fx2 = 0
        self.av_fx = 0
        self.ccf = 0
        self.xi = float(self.indict['xi'])
        self.xi_init = float(self.indict['xi'])
        self.step = int(self.indict['sample_number'])
        self.vect_change = np.zeros(int(self.indict["sample_number"]))

    def _get_space_var(self):
        """
        Returns space variable bounderies (min, max values) for each point
        """
        if self.indict['method'] == 'full_space':
            space_variables = Space(self.indict).full_space()
        elif self.indict['method'] == 'sub_space_C':
            space_variables = Space(self.indict).sub_space_C()
        elif self.indict['method'] == 'sub_space':
            space_variables = Space(self.indict).sub_space()

        return space_variables


    def _get_initial_fx(self, points):
        if int(self.indict['n_processes']) == 1:
            for i in range(len(points)):
                point = points[i]
                fx = Function(self.train_data, self.indict).f_x(point)
                self.fx1[i] = fx
        elif int(self.indict['n_processes']) > 1:

            pool = Pool(processes=int(self.indict['n_processes']))
            results = pool.map(Function(self.train_data, self.indict).f_x, points)
            for i in range(len(results)):
                self.fx1[i]=results[i] 
            pool.close()
            pool.join()
        elif int(self.indict['n_processes']) == -1:
            pool = Pool(processes=os.cpu_count())
            results = pool.map(Function(self.train_data, self.indict).f_x, points)
            for i in range(len(results)):
                self.fx1[i]=results[i]
            pool.close()
            pool.join()


    def _get_stats_initial(self):
        self.av_del_fx  = np.average(self.fx1)
        self.std_fx = np.std(self.fx1)

    def _get_stats(self): 
        self.del_fx = (self.fx2 - self.fx1)**2
        self.av_del_fx  = np.average(self.del_fx)
        self.std_fx = np.std(self.del_fx)

    def _get_vect_change(self, x1, x2):
        #x1 = x1/np.linalg.norm(x1)
        #x2 = x2/np.linalg.norm(x2)

        delX = np.subtract(x1, x2)
        vect_mag = np.linalg.norm(delX)

        return vect_mag

    def _check_convergence(self):

        cond1 = 'NO'
        cond2 = 'NO'
        cond3 = 'NO'

        if self.av_fx <= float(self.indict['conv_fx']):
            cond1 = 'YES'
        if self.av_del_fx <= float(self.indict['conv_del_fx']):
            cond2 = 'YES'
        if self.ccf <= float(self.indict['conv_vec']):
            cond3 = 'YES'
        print_loop_conv(cond1, cond2, cond3)

        if cond1 == cond2 == cond3 == 'YES':
            print('Converged on all three criteria.')
            return 0
        else:
            return 1


    def run(self):
        """
        1. Take sample point or FULL space boundaries 

        2. Generate random sample points within the boundaries of xi or full_space (Latin hypercube method)

        3. Itterate through the points and optimise them with GA or ls
         method using xi 
        """

        program_start = time.time()

        print_logo()
        if int(self.indict['n_processes']) != -1:
            print(f'Number of processes set to {self.indict["n_processes"]}.')
        else:
            print(f'Number of processes set to (-1) {os.cpu_count()}.')

        if self.indict["print_parameters"] == "True":
            print_pars(self.indict)
        
        if os.path.exists(self.indict["out_dir"]):
            shutil.rmtree(self.indict["out_dir"])
        os.makedirs(self.indict["out_dir"])

        train_size = len(self.train_data)


        if self.indict["init_data_sampling"] == 'LHS':
            variable_bounderies = self._get_space_var()

            #Latin hypercube sampling
            if self.indict['random_seed'] != None:
                self.indict['random_seed'] = int((self.indict['random_seed']))

            sampling = LHS(xlimits=variable_bounderies, random_state=self.indict['random_seed'])
            points = sampling(int(self.indict["sample_number"]))

        elif self.indict["init_data_sampling"] == 'void':
            #VOID exploration algorithm
            sampling = VOID(self.indict, self.train_data)
            points  = sampling.search()
        elif self.indict["init_data_sampling"] == 'restart':
            #Read in data points for restart of calculation:
            f = self.indict["restart_file_name"]
            points = np.genfromtxt(f, delimiter=',', dtype=float)
            self.indict['sample_number'] = len(points)
            print(f'Restart data taken from {f} file.')

        #If True: does not combine points with reference data, hence f(x) is calculated only to the respect to ref data
        #and not generated sample points allowing to map function. 
        if self.indict['map_function'] == 'False':
            self.train_data = np.vstack((self.train_data, points))
            

       
        self._get_initial_fx(points)
        self._get_stats_initial()

        #INITIAL DATA!!!! 

        if self.indict['write_initial'] == 'True':
            np.savetxt(f"{self.indict['out_dir']}/initial_points.csv", points, delimiter=",")
        if self.indict['write_initial'] == 'True':    
            np.savetxt(f"{self.indict['out_dir']}/initial_fx.csv", self.fx1, delimiter=",")

        #Calculate initial f(x) and STD for samples
        if int(self.indict["itteration_num"]) != 0:
            init_info(self.av_del_fx, self.std_fx, len(points))

        for itt in range(int(self.indict["itteration_num"])):
            start_time_loop = time.time()
            
            for ix in range(int(self.indict['sample_number'])):
                point_idx = ix + train_size
                point = self.train_data[point_idx]
                
                #print(np.where(self.train_data == point))

                #generate point boundaries
                point_bounderies = Space(self.indict)._sub_space_xi(point, self.xi)
                

                if self.indict["OPT_method"] == "GA":
                    optimised_point_dict = CSPX_GA(self.indict, self.train_data).run_GA(point_bounderies)
                    optimised_point = optimised_point_dict['variable']
                    f_x = optimised_point_dict['function']

                elif self.indict["OPT_method"] == "GRID":
                    optimised = CSPX_GRID(self.indict, self.train_data).run_cspx_grid(point_bounderies)
                    optimised_point = optimised[0]
                    f_x = optimised[1]

                elif self.indict["OPT_method"] == "BO":
                    optimised = CSPX_BO(self.indict, self.train_data).run_bayassian(point_bounderies)
                    optimised_point = optimised[0]
                    f_x = optimised[1]
                    
                else:
                    print('WRONG optimisation method specified!')
                    break

                x1 = point #x1 and x2 for vector diff.
                x2 = optimised_point
                self.vect_change[ix] = self._get_vect_change(x1, x2)


                self.fx2[ix] = f_x
                points[ix] = optimised_point
                self.train_data[point_idx] = optimised_point

            self._get_stats()
            #ccf - coordinate change factor
            self.ccf = np.average(self.vect_change)
            self.av_fx = np.average(self.fx1)

            end_time_loop = time.time()
            loop_time = end_time_loop - start_time_loop 

            #Calculation of the derrivative of f(x)

            self.der_fx2= self.av_del_fx
            der_fx = self.der_fx1 - self.der_fx2
             
            self.der_fx1 = self.der_fx2

            if itt == 0:
                der_fx = 0

            print_loop_info(itt+1, self.av_fx, self.av_del_fx, self.ccf, loop_time, self.indict["OPT_method"], self.indict["print_every"])
            self.fx1 = self.fx2
            self.fx2 = np.zeros(int(self.indict["sample_number"]))
            
            if (itt+1) % int(self.indict['write_f_every']) == 0:
                np.savetxt(f'{self.indict["out_dir"]}/itteration_{itt+1}.csv', points, delimiter=",")
                #PCA reduction
                if self.indict['PCA'] == 'True':
                    PCA(f'{self.indict["out_dir"]}/itteration_{itt+1}.csv').reduce()

            #Writes out stats data:
            #Average derrivative of f(x), average of f(x), 2nd derrivative of average of f(x), std of average f(x), and loop time
            #---------------------------  ---------------  ---------------   ----------------  -------------------      ---------

            if itt == 0:
                fx_header = np.array([['itteration','average of f(x)', 'Average derrivative of f(x)', '2nd derrivative of average of f(x)', 'std of average f(x)',
                'average CCC','loop time']])
                fx_data = np.array([[itt+1, self.av_fx, self.av_del_fx, der_fx, self.std_fx, self.ccf, loop_time]])

                f = open(f'{self.indict["out_dir"]}/fx_data.csv', 'a')
                np.savetxt(f, fx_header, delimiter=",", fmt="%s")
                np.savetxt(f, fx_data, delimiter=",")
                f.close()

            else:
                fx_data = np.array([[itt+1,self.av_fx, self.av_del_fx, der_fx, self.std_fx, self.ccf, loop_time]])
                f = open(f'{self.indict["out_dir"]}/fx_data.csv', 'a')
                np.savetxt(f, fx_data, delimiter=",")
                f.close()
            if (itt+1) % int(self.indict['check_conv_every']) == 0:
                convergence = self._check_convergence()
                if convergence != 0:
                 pass
                else:
                  break
            


    

    
            

if __name__ == "__main__":
    start_time = time.time()
    try:
        inpt = sys.argv[1]
    except IndexError:
        print("Input file must be specified!")
        raise SystemExit

    program = CSPX(inpt)
    program.run()
    end_time = time.time()
    run_time = end_time - start_time
    print_finished()
    print(f"Time elapsed {run_time:.5} s")
    
    
