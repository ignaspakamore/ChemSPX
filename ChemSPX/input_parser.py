import sys
class InputParser:
	def __init__(self, inptfle):

		'''
		Contains default values for input.
		'''
		
		self.input = inptfle
		self.dict = {}
		self.default ={
		'print_parameters':'False',
		'init_data_sampling': 'LHS',
		'out_dir': 'OUT',
		'metric':'euclidean',
		'GRID_sample_number':1000,
		'leaf_size':20,
		'mut_prob':0.1, 
		'cross_prob':0.5,
		'parent_po':0.3,
		'elit_ratio': 0.01,
		'crossover_type': 'uniform',  
		'n_processes':1,
		'write_f_every':1,
		'split_value':0.1,
		'max_iteration_without_improv':50,
		'print_every': 1,
		'write_initial': 'False',
		'map_function': 'False',
		'random_seed': None,
		'check_conv_every': 10,
		'power': 1, 
		'k': 'all',
		'h': 0.1, 
		'verbose': 1, 
		'ploop':'False'}

	def _check_indict(self):

		'''
		Check if neccessary parameters are defined in input file
		and if not default values are inserted.
		'''

		important = [
				'OPT_method',
				'init_data_sampling',
				'Apply_BD',
				'UBL',
				'LBL',
				'sample_number',
				'iteration_num',
				'method',
				'xi',
				'f(x)']
		
		try:
			for key in important:
				if key not in self.dict:
					print(f"Input ERROR: {key} mus be defined in program input file!")
					raise SystemExit
		except:
			raise SystemExit
		
		for key, value in self.default.items():
			if key not in self.dict:
				self.dict[key] = value

		for key, value in self.default.items():
			if value == 'true':
				self.dict[key] ='True'
			elif value == 'false':
				self.dict[key] ='False'


	def get(self) -> dict:

		if type(self.input) == str:

			f = open(self.input, 'r')

			for line in f:
				if not line.startswith('#'):
					if not line == '\n':
						key = line.split(' ', 1)[0].strip()
						val = line.split(' ', 1)[1].strip()
						if '#' in val:
							val = val.split('#', 1)[0].strip()
						self.dict[key] = val

		elif type(self.input) == dict:
			for key, element in self.input.items():
				self.dict[key] = element
			
		else:
			print('Could not parse input file. WRONG FORMAT')
			raise SystemExit
		self._check_indict()

		return self.dict



if __name__ == "__main__":
	inpt = InputParser(sys.argv[1])
	inpt.get()
	inpt._check_indict()
