import numpy as np
from sklearn.neighbors import BallTree
#from printing import print_ploop
import multiprocessing
import time
from modules import Space, CSPX_BO, CSPX_GA, CSPX_GRID


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
		Asigns generated data points to box for optimisation
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
					
					self.pspool[box] = np.append(self.pspool[box], point.reshape(1, 2), axis=0)

				else:
					pass
	def _get_vect_change(self, x1, x2):
		#x1 = x1/np.linalg.norm(x1)
		#x2 = x2/np.linalg.norm(x2)

		delX = np.subtract(x1, x2)
		vect_mag = np.linalg.norm(delX)

		return vect_mag


	def _main_loop(self, index):

		data = self.pspool[index]

		sample_number = len(data) - self.last_train_data_index

		vect_change = np.zeros(sample_number)
		fx = np.zeros(sample_number)
	
		for ix in range(sample_number):
			point_idx = ix + self.last_train_data_index 
			point = data[point_idx]
						
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
			vect_change[ix] = self._get_vect_change(x1, x2)

			data[point_idx] = optimised_point

			fx[ix] = f_x

		#Statistics
		#del_fx = (fx2 - fx1)**2
		#av_del_fx  = np.average(del_fx)
		#std_fx = np.std(del_fx)
		#ccf - coordinate change factor
		#ccf = np.average(vect_change)
		#av_fx = np.average(fx1)

	
		return [fx,  vect_change, data]


	def _run_in_paralel(self):
		pool = multiprocessing.Pool()
		pool.map(self._main_loop, range(self.n_processes))

					
	def run(self):
		self._gen_cp_of_space()
		self._divide_space()
		self._group()
		self._run_in_paralel()

		

	
		











