import pandas as pd
import numpy as np
import PySimpleGUI as sg
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d


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
				for i in range(self.data_type.nunique()):
					self.colour[self.data_type.unique()[i]] = data_type_colour[i]

			self.data = self.data.drop('Type', 1)
		else:
			pass


	def plot2D(self):
		
		plt.scatter(self.PincipalComponents[:,0], self.PincipalComponents[:,1], color=[self.colour[r] for r in self.data_type])
		lables = []
	
		for key, val  in self.colour.items():
			lables.append(mpatches.Patch(color=f'{val}', label=f'{key}'))
		
		plt.legend(handles=lables)

		if self.anotate == True:
			for x,y,label in zip(self.PincipalComponents[0],self.PincipalComponents[1], self.data_type):

				plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center') 
				
		plt.show()


	def plot3D(self):
		fig = plt.figure(figsize=(10, 10))

		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(self.PincipalComponents[:,0], self.PincipalComponents[:,1], self.PincipalComponents[:,2], 
			color=[self.colour[r] for r in self.data_type], picker = True) 

		lables = []
	
		for key, val  in self.colour.items():
			lables.append(mpatches.Patch(color=f'{val}', label=f'{key}'))


		plt.legend(handles=lables)
		


		def distance(point, event):
			"""Return distance between mouse position and given data point

			Args:
				point (np.array): np.array of shape (3,), with x,y,z in data coords
				event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
			Returns:
				distance (np.float64): distance (in screen coords) between mouse pos and data point
			"""
			assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

			# Project 3d data space to 2d data space
			x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
			# Convert 2d data space to 2d screen space
			x3, y3 = ax.transData.transform((x2, y2))

			return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


		def calcClosestDatapoint(X, event):
			""""Calculate which data point is closest to the mouse position.

			Args:
				X (np.array) - array of points, of shape (numPoints, 3)
				event (MouseEvent) - mouse event (containing mouse position)
			Returns:
				smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
			"""
			distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
			return np.argmin(distances)


		def annotatePlot(X, index):
			"""Create popover label in 3d chart

			Args:
				X (np.array) - array of points, of shape (numPoints, 3)
				index (int) - index (into points array X) of item which should be printed
			Returns:
				None
			"""
			# If we have previously displayed another label, remove it first
			if hasattr(annotatePlot, 'label'):
				annotatePlot.label.remove()
			# Get data point from array of points X, at position index
			x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
			annotatePlot.label = plt.annotate( "Index %d" % index,
				xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
				arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
			fig.canvas.draw()


		def onMouseMotion(event):
			"""Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
			closestIndex = calcClosestDatapoint(self.PincipalComponents, event)
			annotatePlot (self.PincipalComponents, closestIndex)
		if self.anotate == True:
			fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion

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
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2))],
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
				if values[3] == True:
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




