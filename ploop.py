import numpy as np
from sklearn.neighbors import BallTree
#from printing import print_ploop
from multiprocessing import Pool
import time
from modules import Space


class PLoop():
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

		'''
		Creates a pool of n replicated train_data
		and asigns generated data points for optimisation
		'''

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

		fx1 = np.zeros(int(self.indict["sample_number"]))
		fx2 = np.zeros(int(self.indict["sample_number"]))

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

				self.train_data[point_idx] = optimised_point
				self.fx2[ix] = f_x
			

	def _run_in_paralel(self):
		pass
					
	def run(self):
		self._gen_cp_of_space()
		self._divide_space()
		self._group()
		print (self.pspool)
		











