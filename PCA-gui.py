import pandas as pd
import numpy as np
import PySimpleGUI as sg
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt


class PCAGUI():
	def __init__(self):

		self.file_dir = ''
		self.n_componenets = 0
		self.data = None
		self.colour = {}
		self.PincipalComponents = None


	def get_data(self):
		try:
			self.data = pd.read_csv(self.file_dir)
		except IOError:
			self.data = pd.read_excel(self.file_dir)

		if 'Type' in self.data:

			data_type= self.data['Type']

			if 'Colour' in self.data:
				data_type_colour= self.data['Colour']
				for idx, tpe in enumerate(data_type):
					self.colour[tpe] = data_type_colour[idx]
				self.data = self.data.drop('Colour', 1)
			else:
				data_type_colour = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']

			self.data = self.data.drop('Type', 1)
		else:
			pass

				

	def plot2D(self):
		plt.scatter(self.PincipalComponents.T[0], self.PincipalComponents.T[1], c=self.colour)
		plt.show()
	def plot3D(self):
		pass

	def reduce(self):
		pca = PCA(n_components=self.n_componenets)
		self.PincipalComponents = pca.fit_transform(self.data)

	def run(self):

		sg.theme('Default')   # Add a touch of color
		


		# All the stuff inside your window.
		output = sg.Text()
		layout = [
					[sg.Text('Data file'), sg.InputText()],
					[sg.Text('Number of components')],
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2)), sg.Text('other:'), sg.InputText()],
					[sg.Button('Calculate'),sg.Button('PLOT'), sg.Button('SAVE')],
					[output]]
		

		# Create the Window
		window = sg.Window('CSPX PCA-GUI', layout, size=(350, 150), font=30, finalize=True)
		# Event Loop to process "events" and get the "values" of the inputs

		while True:
			event, values = window.read()

			if event == 'Calculate':
				'''
				Set calculation parameters
				'''
				self.file_dir = str(values[0])
				if values[1] == True:
					self.n_componenets = 2
				elif values[2] == True:
					self.n_componenets = 3
				elif values[3] != '':
					self.n_componenets = int(values[3])

				self.get_data()
				print (self.colour)
				self.reduce()

				output.update('FINISHED √')
				
				
			elif event == 'PLOT':
				self.plot2D()
			elif event == 'SAVE':
				pass
			elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
				break




		window.close()




if __name__ == '__main__':
	program = PCAGUI()
	program.run()




