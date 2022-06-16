import numpy as np
from sklearn.neighbors import BallTree
#from printing import print_ploop
from multiprocessing import Pool


class PLoop():
	"""docstring for PLoop"""
	def __init__(self, indict):
		self.indict = indict
		self.n_processes = 3
		self.resolution = 1+self.n_processe
		self.pspool = []

	def _gen_cp_of_space():
		for i in range(self.n_processes):
			pspool.append(np.copy(self.train_data))

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
		u = self.indict["UBL"]
		l = self.indict["LBL"]

		max_val = np.fromstring(u, sep=',')
		min_val = np.fromstring(l, sep=',')


		axes = np.zeros((len(max_val), self.resolution))
		split_axes = np.zeros((self.n_processes, len(max_val), 2))

		for i in range(len(max_val)):
			values = self._gen_array(max_val[i], min_val[i])
			for j, val in enumerate(values):
				axes[i][j] = val

		for i, a in enumerate(axes):
			split = self._split_array(a)
			for j in range(len(split)):
				split_axes[j][i][0] = split[j][0][0]
				split_axes[j][i][1] = split[j][0][1]

		return split_axes		


	def _group(self):
		pass

	def run(self):
		pass
