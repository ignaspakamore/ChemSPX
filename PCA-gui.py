import pandas as pd
import numpy as np
import PySimpleGUI as sg
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class PCAGUI():
	def __init__(self):

		self.file_dir = ''
		self.n_componenets = 0
		self.data = None
		self.colour = {}
		self.PincipalComponents = None
		self.data_type = None
		self.ERROR = ''
		self.anotate = False


	def open_file(self):
	
		if self.file_dir.endswith('.csv'):
			self.data = pd.read_csv(self.file_dir)
		elif self.file_dir.endswith('.xlsx' ):
			self.data = pd.read_excel(self.file_dir)
		else:
			self.ERROR = 'WRONG FILE FORMAT'

	def get_data(self):
		if 'Type' in self.data:

			self.data_type = self.data['Type']

			if 'Colour' in self.data:
				data_type_colour= self.data['Colour']
				for idx, tpe in enumerate(self.data_type):
					self.colour[tpe] = data_type_colour[idx]
				self.data = self.data.drop('Colour', 1)
			else:
				data_type_colour = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']

			self.data = self.data.drop('Type', 1)
		else:
			pass


	def plot2D(self):
		self.PincipalComponents = self.PincipalComponents.T
		plt.scatter(self.PincipalComponents[0], self.PincipalComponents[1], color=[self.colour[r] for r in self.data_type])
		lables = []
	
		for key, val  in self.colour.items():
			lables.append(mpatches.Patch(color=f'{val}', label=f'{key}'))
		
		plt.legend(handles=lables)

		if self.anotate == True:
			for x,y,label in zip(self.PincipalComponents[0],self.PincipalComponents[1], self.data_type):

				plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center') 
				
		plt.show()
	def plot3D(self):
		fig = plt.figure(figsize=(4,4))

		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(self.PincipalComponents[0], self.PincipalComponents[1], self.PincipalComponents[2]) # plot the point (2,3,4) on the figure

		plt.show()

	def reduce(self):
		pca = PCA(n_components=self.n_componenets)
		self.PincipalComponents = pca.fit_transform(self.data)

	def run(self):

		sg.theme('DefaultNoMoreNagging')   # Add a touch of color
		


		# All the stuff inside your window.
		output = sg.Text()
		layout = [
					[sg.Text('Data file'), sg.In(size=(35,1), enable_events=True), sg.FileBrowse()],
					[sg.Text('Number of components')],
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2)), sg.Text('other:'), sg.InputText(size=(3,1))],
					[sg.Checkbox('Anotate samples',size = (20,2))],
					[sg.Button('Calculate'),sg.Button('PLOT'), sg.Button('SAVE')],
					[output]]
		

		# Create the Window
		window = sg.Window('CSPX PCA-GUI', layout, size=(600, 300), font=30, finalize=True)
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

				self.open_file()
				output.update(self.ERROR)

				if self.ERROR == '':
					self.get_data()
					self.reduce()
					output.update('FINISHED √')
				self.ERROR = ''
				
			elif event == 'PLOT':
				if values[4] == True:
					self.anotate = True
				if self.n_componenets == 2:
					self.plot2D()
				elif self.n_componenets == 3:
					self.plot3D()
				else:
					output.update('PCA must be calculated first')
				self.anotate = False

			elif event == 'SAVE':
				pass
			elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
				break


		window.close()




if __name__ == '__main__':
	program = PCAGUI()
	program.run()




