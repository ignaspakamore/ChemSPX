import PySimpleGUI as sg
import pandas as pd
import numpy as np
from pca import PCA





class PCAGUI():
	def __init__(self):

		self.file_dir = ''
		self.n_componenets = 0
		self.data = None
		self.data_type = None
		self.colour = None
		self.pca = None


	def get_data(self):
		try:
			data = pd.read_csv(file)
		except IOError:
			data = pd.read_excle(file)

		self.data_type= data['Type']
		

	def plot2D(self):
		pass
	def plot3D(self):
		pass

	def reduce(self):
		pass



	def run(self):

		sg.theme('Default')   # Add a touch of color

		# All the stuff inside your window.
		layout = [
					[sg.Text('Data file'), sg.InputText()],
					[sg.Text('Number of components')],
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2)), sg.Text('other:'), sg.InputText()],
					[sg.Button('Calculate'),sg.Button('PLOT'), sg.Button('SAVE')] ]

		# Create the Window
		window = sg.Window('CSPX PCA-GUI', layout, size=(350, 150), font=30)
		# Event Loop to process "events" and get the "values" of the inputs
		while True:
			event, values = window.read()

			if event == 'Calculate':
				'''
				Set calculation parameters
				'''
				self.data = values[0]
				if values[1] == True:
					self.n_componenets = 2
				elif values[2] == True:
					self.n_componenets = 3
				elif values[3] != '':
					self.n_componenets = int(values[3])
				

			elif event == 'PLOT':
				pass
			elif event == 'SAVE':
				pass
			elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
				break




		window.close()




if __name__ == '__main__':
	program = PCAGUI()
	program.run()




