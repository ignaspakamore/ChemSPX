import numpy as np
from sklearn.neighbors import BallTree
#from printing import print_ploop
from multiprocessing import Pool
import time
from modules import Space
from cspx import CSXP


class PLoop(CSPX):
	"""docstring for PLoop"""
	def __init__(self, indict, train_data):
		self.indict = indict
		self.train_data = train_data
		self.last_train_data_index = len(self.train_data) - int(self.indict['sample_number']) -1
		self.n_processes = 3
		self.resolution = 1+self.n_processes
		self.pspool = []
		self.split_axes =np.zeros((self.n_processes, len(np.fromstring(self.indict['UBL'], sep=',')), 2))
		self.xi = float(self.indict['xi'])

	def _gen_cp_of_space(self):
		for i in range(self.n_processes):
			self.pspool.append(np.copy(self.train_data[0:self.last_train_data_index]))

	def _gen_array(self, max_val, min_val):
		array = np.zeros(self.resolution)
		spacing = max_val/self.resolution
		for i in range(self.resolution):
			if i == 0:
				array[i] = min_val
			else:
				min_val += spacing
				array[i] = min_val

		array[self.resolution-1] = max_val
		
		return array
	def _split_array(self, array):

		split = np.zeros((self.n_processes,1, 2))

		for i in range(self.n_processes):
			split[i][0] = array[i]
			split[i][0][1] = array[i+1]

		return split
		
	def _divide_space(self):
		'''
		Space is divided into 'boxes'. 
		access: self.split_axes[BOX][PARAMETER][UPPER:LOWER]
		'''
		u = self.indict["UBL"]
		l = self.indict["LBL"]

		max_val = np.fromstring(u, sep=',')
		min_val = np.fromstring(l, sep=',')


		axes = np.zeros((len(max_val), self.resolution))

		for i in range(len(max_val)):
			values = self._gen_array(max_val[i], min_val[i])
			for j, val in enumerate(values):
				axes[i][j] = val

		for i, a in enumerate(axes):
			split = self._split_array(a)
			for j in range(len(split)):
				self.split_axes[j][i][0] = split[j][0][0]
				self.split_axes[j][i][1] = split[j][0][1]

	def _group(self):

		points = self.train_data[self.last_train_data_index:]

		for point in points:
			for box in range(len(self.split_axes)):
				for i, val in enumerate(point):
					accept = []
					if val >= self.split_axes[box][i][0] and val <=self.split_axes[box][i][1]:
						accept.append(True)
					else:
						accept.append(False)

				if all(accept) == True:
					self.pspool[box] = np.append(self.pspool[box], point)
				else:
					pass
	def _main_loop(self):


		for itt in range(int(self.indict["itteration_num"])):
			start_time_loop = time.time()

			for ix in range(int(self.indict['sample_number'])):
				point_idx = ix + self.last_train_data_index +1
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
				np.savetxt(f'{self.indict["out_dir"]}/itteration_{itt+1}.csv', self.train_data[self.train_size:len(self.train_data)], delimiter=",")
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


	def _run_in_paralel(self):
		pass
					
	def run(self):
		self._gen_cp_of_space()
		self._divide_space()
		self._group()
		











