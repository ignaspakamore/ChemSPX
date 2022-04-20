import PySimpleGUI as sg
import _thread 


class CSPXGUI():
	def __init__(self):
		self.indict = None
		self.stop = False

	def prnt(self):

		for i in range(1000):
			print ('Hi')

	def run(self):

		sg.theme('DefaultNoMoreNagging')   # Add a touch of color

		# All the stuff inside your window.
		layout = [[sg.Text('Data file'), sg.InputText()],
					[sg.Text('Number of components')],
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2)), sg.Text('other:'), sg.InputText()],
					[sg.Button('RUN'),sg.Button('Generate input')],
					[sg.Button('STOP'),sg.Button('Generate input')],
					[sg.Txt('', size=(8,1), key='output')]]

		# Create the Window
		window = sg.Window('CSPX-GUI', layout, size=(350, 150), font=30)
		# Event Loop to process "events" and get the "values" of the inputs
		while True:
			event, values = window.read()
			if event == 'RUN':
				self.prnt()

			if event == 'STOP':
				


		window.close()


if __name__ == "__main__":
	program = CSPXGUI()

	program.run()



