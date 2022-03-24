from smt.sampling_methods import LHS
from modules import CSPX_GA, CSPX_GRID, CSPX_BO, VOID, Space, Function
from input_parser import InputParser


class LHS_EQ():
	def __init__(self, indict):
		self.indict = indict
		self.samples = Space(self.indict).full_space()

	def equilibrate(self):

		for _ in range(len(self.indict['EQ_steps'])):
			for i in range(len(self.samples)):
				
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

